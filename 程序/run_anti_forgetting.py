import argparse
import csv
import json
import platform
import time
from pathlib import Path

import torch
from torch import nn
from torch.func import functional_call, grad, vmap
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from run_cross_dataset_sequence import (
    CLASS_NAMES, OUTPUT, TASKS, batchnorm_state, evaluate, evaluate_configs,
    load_task, loader, seed_everything, write_json,
)
from wdcnn import WDCNN


STAGES = ["initial_A", "after_B", "after_C", "after_D"]
METHODS = ("sequential", "ewc", "pq_replay", "uciln")
SUBSPACES = 16
CENTERS = 32
BUFFER_PER_DOMAIN = 512
MMD_WEIGHT = 1.0
SIGMA_FACTORS = (0.25, 0.5, 1.0, 2.0, 4.0)


class ProductQuantizer:
    def __init__(self, features, rng):
        points = self.split(features)
        indices = torch.randperm(len(features), generator=rng)[:CENTERS]
        centers = points[:, indices].contiguous()
        for _ in range(20):
            indices = torch.cdist(points, centers).argmin(dim=-1)
            memberships = F.one_hot(indices, CENTERS).to(points.dtype).transpose(1, 2)
            counts = memberships.sum(dim=-1, keepdim=True)
            centers = torch.where(counts > 0, memberships @ points / counts.clamp_min(1), centers)
        self.centers = centers

    @staticmethod
    def split(features):
        return features.reshape(len(features), SUBSPACES, -1).transpose(0, 1).contiguous()

    def encode(self, features):
        parts = self.split(features)
        return torch.cdist(parts, self.centers).argmin(dim=-1).T.to(torch.uint8)

    def decode(self, codes):
        parts = self.centers[torch.arange(SUBSPACES, device=codes.device), codes.long()]
        return parts.flatten(1)


def features(model, inputs, device):
    model.features.eval()
    with torch.no_grad():
        return torch.cat([model.features(batch.to(device).unsqueeze(1)).flatten(1)
                          for batch in inputs.split(128)])


def mmd(source, target):
    joined = torch.cat((source, target))
    squared = torch.cdist(joined, joined).square()
    bandwidth = squared.detach().sum() / (len(joined) * (len(joined) - 1))
    factors = squared.new_tensor(SIGMA_FACTORS)[:, None, None]
    kernel = (-squared / (bandwidth.clamp_min(1e-6) * factors)).exp().mean(dim=0)
    n_source = len(source)
    return (kernel[:n_source, :n_source].mean() + kernel[n_source:, n_source:].mean()
            - 2 * kernel[:n_source, n_source:].mean())


def index_batches(size, batch_size, rng, steps, track=None):
    n_batches = size // batch_size
    if not n_batches:
        raise ValueError(f"Cannot draw a full batch of {batch_size} from {size} windows")
    iterator = DataLoader(torch.arange(size), batch_size=batch_size, shuffle=True,
                          drop_last=True, generator=rng)
    remaining = steps
    while remaining:
        drawn = min(remaining, n_batches)
        if track is not None:
            track["tail_dropped"] += size % batch_size if drawn == n_batches else 0
            track["budget_skipped"] += size - drawn * batch_size if drawn < n_batches else 0
        for _, indices in zip(range(drawn), iterator):
            yield indices
        remaining -= drawn


def sample(codes, count, rng):
    return torch.randint(len(codes), (count,), generator=rng).to(codes.device)


def keep_codes(codes, rng, labels=None):
    if labels is None or len(codes) <= BUFFER_PER_DOMAIN:
        return torch.randperm(len(codes), generator=rng)[:BUFFER_PER_DOMAIN].to(codes.device)
    population = labels.bincount(minlength=len(CLASS_NAMES)).tolist()
    while sum(population) > BUFFER_PER_DOMAIN:
        largest = max(range(len(population)), key=population.__getitem__)
        population[largest] -= 1
    selection = []
    for class_id, count in enumerate(population):
        members = (labels == class_id).nonzero().flatten()
        selection.append(members[torch.randperm(len(members), generator=rng)[:count]])
    return torch.cat(selection)


def fisher_diagonal(model, train, device, rng, limit):
    model.eval()
    params = {name: value.detach().clone() for name, value in model.named_parameters()}
    buffers = dict(model.named_buffers())
    inputs, labels = train
    chosen = torch.randperm(len(inputs), generator=rng)[:limit]
    fisher = {name: torch.zeros_like(value) for name, value in params.items()}

    def loss(weights, signal, label):
        logits = functional_call(model, (weights, buffers), (signal[None, :],))
        return F.cross_entropy(logits, label[None])

    for batch in chosen.split(16):
        gradients = vmap(grad(loss), in_dims=(None, 0, 0))(
            params, inputs[batch].to(device), labels[batch].to(device),
        )
        for name, gradient in gradients.items():
            fisher[name].add_(gradient.detach().square().sum(dim=0) / len(chosen))
    return {"anchor": params, "fisher": fisher, "samples": len(chosen), "indices": chosen}


def supervised_stage(model, train, device, args, seed, memories):
    inputs, labels = (part.to(device) for part in train)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = torch.Generator().manual_seed(seed)
    used = torch.zeros(len(inputs), dtype=torch.bool)
    budget = {"tail_dropped": 0, "budget_skipped": 0, "source_replay_draws": 0,
              "target_replay_draws": 0, "supervised": True, "target_windows_per_update": args.batch_size}
    model.train()
    for indices in index_batches(len(inputs), args.batch_size, rng, args.steps, budget):
        used[indices] = True
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs[indices])
        classification = F.cross_entropy(logits, labels[indices])
        penalty = classification.new_zeros(())
        for memory in memories:
            penalty = penalty + sum(
                (memory["fisher"][name] * (value - memory["anchor"][name]).square()).sum()
                for name, value in model.named_parameters()
            )
        regularization = args.ewc_lambda * penalty / 2
        loss = classification + regularization
        loss.backward()
        optimizer.step()
        yield {"loss": loss.item(), "ce": classification.item(), "mmd": 0.0,
               "regularization": regularization.item()}
    budget["unique_windows_used"] = int(used.sum())
    yield budget


def replay_stage(model, target, pq, target_pq, source_memory, previous_targets, device,
                 args, seed, align, current_source=None):
    model.features.eval().requires_grad_(False)
    model.classifier.eval()
    rng = torch.Generator().manual_seed(seed)
    replay_rng = torch.Generator().manual_seed(seed + 40_000)
    target_replay_rng = torch.Generator().manual_seed(seed + 60_000)
    source_codes, source_labels = source_memory
    target_codes = target_pq.encode(features(model, target, device)) if align else None
    history = torch.cat(previous_targets) if previous_targets else None
    if current_source is not None:
        current_inputs, current_labels = current_source
        current_codes = pq.encode(features(model, current_inputs, device))
        source_batches = index_batches(len(current_codes), args.batch_size, replay_rng, args.steps)
        current_labels = current_labels.to(device)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr)
    used = torch.zeros(len(target), dtype=torch.bool)
    budget = {"tail_dropped": 0, "budget_skipped": 0, "source_replay_draws": 0,
              "target_replay_draws": 0, "supervised": False,
              "target_windows_per_update": args.batch_size if align else 0}
    norm_training = {name: value.detach().clone() for name, value in model.features.named_buffers()}
    target_batches = (index_batches(len(target), args.batch_size, rng, args.steps, budget)
                      if align else (None for _ in range(args.steps)))
    for indices in target_batches:
        replay_indices = sample(source_codes, args.batch_size, replay_rng)
        train_codes, train_labels = source_codes[replay_indices], source_labels[replay_indices]
        if current_source is not None:
            new_indices = next(source_batches)
            train_codes = torch.cat((current_codes[new_indices], train_codes))
            train_labels = torch.cat((current_labels[new_indices], train_labels))
        source_features = pq.decode(train_codes)
        optimizer.zero_grad(set_to_none=True)
        source_logits = model.classifier(source_features)
        classification = F.cross_entropy(source_logits, train_labels)
        alignment = classification.new_zeros(())
        if align:
            used[indices] = True
            batch_codes = target_codes[indices]
            if history is not None:
                batch_codes = torch.cat((batch_codes, history[sample(history, args.batch_size, target_replay_rng)]))
                budget["target_replay_draws"] += args.batch_size
            alignment = mmd(source_logits, model.classifier(target_pq.decode(batch_codes)))
        loss = classification + MMD_WEIGHT * alignment
        loss.backward()
        optimizer.step()
        budget["source_replay_draws"] += args.batch_size
        yield {"loss": loss.item(), "ce": classification.item(), "mmd": alignment.item(),
               "regularization": 0.0}
    for name, value in model.features.named_buffers():
        if not torch.equal(value, norm_training[name]):
            raise RuntimeError(f"Frozen feature buffer changed: {name}")
    budget["unique_windows_used"] = int(used.sum()) if align else 0
    retained = source_codes[:0]
    if align:
        seen_codes = target_codes[used]
        retained = seen_codes[keep_codes(seen_codes, replay_rng)]
    yield budget, retained


def record_steps(generator, writer, stream, seed, method, stage):
    started = time.monotonic()
    last = next(generator)
    for step, following in enumerate(generator, start=1):
        writer.writerow({"seed": seed, "method": method, "stage": stage, "step": step,
                         "elapsed_seconds": round(time.monotonic() - started, 3), **last})
        if step % 20 == 0:
            stream.flush()
        last = following
    stream.flush()
    return last


def run_sequence(method, seed, model, datasets, tests, validations, baseline, output, device, args):
    folder = output / method / str(seed)
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(baseline / str(seed) / "wdcnn_initial_A.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    profiles = dict(checkpoint["domain_bn"])
    scores, a_shared, validation, budgets = {}, {}, {}, {}
    scores[STAGES[0]], a_shared[STAGES[0]] = evaluate_configs(model, tests, device, profiles)
    validation[STAGES[0]], _ = evaluate_configs(model, validations, device, profiles)
    expected = json.loads((baseline / str(seed) / "stage_metrics.json").read_text())["scores"][STAGES[0]][TASKS[0]]
    if scores[STAGES[0]][TASKS[0]]["confusion_matrix"] != expected["confusion_matrix"]:
        raise RuntimeError(f"A checkpoint does not reproduce the initial A confusion matrix for {seed}")
    rng = torch.Generator().manual_seed(seed + 20_000)
    memories = []
    target_memories = []
    method_state = {}
    if method == "ewc":
        memories.append(fisher_diagonal(model, datasets[TASKS[0]]["train"], device, rng, args.fisher_samples))
        method_state["ewc_memories"] = memories
    if method in ("uciln", "pq_replay"):
        source = datasets[TASKS[0]]["train"]
        source_features = features(model, source[0], device)
        pq = ProductQuantizer(source_features, rng)
        source_codes = pq.encode(source_features)
        source_labels = source[1].to(device)
        kept = keep_codes(source_codes, rng, source_labels)
        source_memory = source_codes[kept], source_labels[kept]
        method_state = {"codebook": pq.centers, "source_memory": source_memory,
                        "target_memory": target_memories}
    torch.save({**checkpoint, **method_state}, folder / "wdcnn_initial_A.pt")
    fields = ["seed", "method", "stage", "step", "loss", "ce", "mmd", "regularization", "elapsed_seconds"]
    with (folder / "training_log.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for stage, task in zip(STAGES[1:], TASKS[1:]):
            seed_everything(seed)
            started = time.monotonic()
            if method in ("sequential", "ewc"):
                train = datasets[task]["train"]
                loop = supervised_stage(model, train, device, args, seed, memories)
                budget = record_steps(loop, writer, stream, seed, method, stage)
                if method == "ewc":
                    memories.append(fisher_diagonal(model, train, device, rng, args.fisher_samples))
            else:
                # Unlabeled arm receives only the target input tensor.
                loop = replay_stage(model, datasets[task]["train"][0], pq, pq, source_memory,
                                    target_memories, device, args, seed, method == "uciln")
                budget, new_memory = record_steps(loop, writer, stream, seed, method, stage)
                if len(new_memory):
                    target_memories.append(new_memory)
            budget.update({"update_steps": args.steps, "train_windows": len(datasets[task]["train"][0]),
                           "adaptation_seconds": round(time.monotonic() - started, 2)})
            if method == "pq_replay":
                budget.update(tail_dropped=0, budget_skipped=0)
            budgets[stage] = budget
            profiles[task] = batchnorm_state(model)
            scores[stage], a_shared[stage] = evaluate_configs(model, tests, device, profiles)
            validation[stage], _ = evaluate_configs(model, validations, device, profiles)
            torch.save({"model": model.state_dict(), "domain_bn": profiles, "classes": CLASS_NAMES,
                        "stage": stage, **method_state}, folder / f"wdcnn_{stage}.pt")
            print(f"{method} seed={seed} {stage} ({budget['adaptation_seconds']}s) "
                  + " ".join(f"{name}={value['accuracy']:.2%}" for name, value in scores[stage].items()), flush=True)
    result = {"scores": scores, "a_with_shared_bn": a_shared, "validation": validation,
              "training_budgets": budgets}
    write_json(folder / "stage_metrics.json", result)
    return result


def mini_protocol(datasets, args, output, device):
    output.mkdir(parents=True, exist_ok=True)
    seed = args.seeds[0]
    seed_everything(seed)
    rng = torch.Generator().manual_seed(seed)
    source, target = (datasets[task] for task in TASKS[:2])

    def phase(split, new):
        selected = (split[1] == 2) if new else (split[1] < 2)
        return tuple(tensor[selected] for tensor in split)

    model = WDCNN(num_classes=len(CLASS_NAMES)).to(device)
    source_initial = phase(source["train"], False)
    target_initial = phase(target["train"], False)[0]
    source_batches = index_batches(len(source_initial[0]), args.batch_size, rng, args.steps)
    target_batches = index_batches(len(target_initial), args.batch_size, rng, args.steps)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    with (output / "training_log.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["seed", "method", "stage", "step", "loss", "ce", "mmd", "regularization", "elapsed_seconds"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        started = time.monotonic()
        for step, (src, tgt) in enumerate(zip(source_batches, target_batches), start=1):
            optimizer.zero_grad(set_to_none=True)
            joined = torch.cat((source_initial[0][src], target_initial[tgt])).to(device).unsqueeze(1)
            zs, zt = model.features(joined).flatten(1).chunk(2)
            logits = model.classifier(zs)[:, :2]
            classification = F.cross_entropy(logits, source_initial[1][src].to(device))
            alignment = mmd(zs, zt)
            loss = classification + MMD_WEIGHT * alignment
            loss.backward()
            optimizer.step()
            writer.writerow({"seed": seed, "method": "uciln_protocol", "stage": "phase_0", "step": step,
                             "loss": loss.item(), "ce": classification.item(), "mmd": alignment.item(),
                             "regularization": 0.0, "elapsed_seconds": round(time.monotonic() - started, 3)})
        source_initial_features = features(model, source_initial[0], device)
        target_initial_features = features(model, target_initial, device)
        pq = ProductQuantizer(source_initial_features, rng)
        target_pq = ProductQuantizer(target_initial_features, rng)
        source_codes = pq.encode(source_initial_features)
        source_labels = source_initial[1].to(device)
        chosen = keep_codes(source_codes, rng, source_labels)
        source_memory = source_codes[chosen], source_labels[chosen]
        target_codes = target_pq.encode(target_initial_features)
        target_memory = target_codes[keep_codes(target_codes, rng)]
        test = loader(target, "test", args.batch_size, device)

        class PhaseZero(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, inputs):
                logits = self.model(inputs).clone()
                logits[:, 2] = torch.finfo(logits.dtype).min
                return logits

        old_test = loader({"test": phase(target["test"], False)}, "test", args.batch_size, device)
        new_test = loader({"test": phase(target["test"], True)}, "test", args.batch_size, device)
        initial = evaluate(PhaseZero(model), old_test, device)
        state = {"classes": CLASS_NAMES, "codebook": pq.centers, "target_codebook": target_pq.centers,
                 "source_memory": source_memory, "target_memory": [target_memory]}
        torch.save({"model": model.state_dict(), "active_class_ids": [0, 1], **state}, output / "phase_0.pt")
        new_source = phase(source["train"], True)
        loop = replay_stage(model, phase(target["train"], True)[0], pq, target_pq, source_memory,
                            [target_memory], device, args, seed, True, current_source=new_source)
        budget, new_memory = record_steps(loop, writer, stream, seed, "uciln_protocol", "phase_1")
        # Commit new labeled source codes and target codes to their bounded buffers.
        combined_codes = torch.cat((source_memory[0], pq.encode(features(model, new_source[0], device))))
        combined_labels = torch.cat((source_memory[1], new_source[1].to(device)))
        chosen = keep_codes(combined_codes, rng, combined_labels)
        combined_targets = torch.cat((target_memory, new_memory))
        torch.save({"model": model.state_dict(), **state, "active_class_ids": [0, 1, 2],
                    "source_memory": (combined_codes[chosen], combined_labels[chosen]),
                    "target_memory": [combined_targets[keep_codes(combined_targets, rng)]]}, output / "phase_1.pt")
        after_old = evaluate(model, old_test, device)
        after_new = evaluate(model, new_test, device)
        after_all = evaluate(model, test, device)
    write_json(output / "stage_metrics.json", {"seed": seed, "phase_0_old": initial,
               "phase_1_old": after_old, "phase_1_new": after_new, "phase_1_all": after_all,
               "phase_1_budget": budget})
    print(f"UCILN class-incremental protocol old {initial['accuracy']:.2%} → {after_old['accuracy']:.2%}; "
          f"new {after_new['accuracy']:.2%}; overall {after_all['accuracy']:.2%}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=OUTPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT.parent / "anti_forgetting")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--ewc-lambda", type=float, default=1000.0)
    parser.add_argument("--fisher-samples", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    cpu_info = Path("/proc/cpuinfo")
    cpu_name = next((line.split(":", 1)[1].strip() for line in cpu_info.read_text().splitlines()
                     if line.startswith("model name")), platform.processor()) if cpu_info.is_file() else platform.processor()
    write_json(args.output / "run_config.json", {
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cpu": cpu_name, "system": platform.freedesktop_os_release().get("PRETTY_NAME", platform.system()),
        "torch": str(torch.__version__), "python": platform.python_version(), "cpu_threads": torch.get_num_threads(),
        "steps": args.steps, "batch_size": args.batch_size, "lr": args.lr, "seeds": args.seeds,
        "fisher_samples_per_domain": args.fisher_samples, "ewc_lambda": args.ewc_lambda,
        "mmd_weight": MMD_WEIGHT, "gaussian_bandwidth_factors": SIGMA_FACTORS,
        "pq_subspaces": SUBSPACES, "pq_centers_per_subspace": CENTERS, "memory_per_domain": BUFFER_PER_DOMAIN,
        "initial_checkpoints": args.baseline.name, "methods": METHODS,
    })
    datasets, manifests = {}, {}
    for task in TASKS:
        datasets[task], manifests[task] = load_task(task)
        print(f"{task}: {manifests[task]['counts']}", flush=True)
    write_json(args.output / "split_manifests.json", manifests)
    tests = {task: loader(datasets[task], "test", args.batch_size, device) for task in TASKS}
    validations = {task: loader(datasets[task], "validation", args.batch_size, device) for task in TASKS}
    mini_protocol(datasets, args, args.output / "uciln_class_incremental", device)
    results = {}
    for method in METHODS:
        results[method] = {}
        for seed in args.seeds:
            seed_everything(seed)
            model = WDCNN(num_classes=len(CLASS_NAMES)).to(device)
            results[method][seed] = run_sequence(method, seed, model, datasets, tests, validations,
                                                 args.baseline, args.output, device, args)
        write_json(args.output / "stage_metrics.json", results)


if __name__ == "__main__":
    main()
