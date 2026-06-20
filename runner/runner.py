import argparse
import asyncio
import json
import os
import queue
import re
import sys
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

from diskcache import Cache
from omegaconf import OmegaConf

sys.path.append(os.getcwd())

from cachesaver.pipelines import OnlineAPI
from src import EnvironmentFactory, MethodFactory
from src.methods import *
from src.models import API, OnlineLLM
from src.tasks import *
from src.tasks.benchmark_rw.benchmark import BenchmarkBenchmarkRW
from src.typedefs import DecodingParameters


RUNNER_NAME = "runner"
RUNNER_METHOD = "reagents"
RUNNER_BENCHMARK = "benchmark_rw"
RUNNER_SPLIT = "single"


class ReAgentsCanvas(tk.Canvas):
    COLORS = {
        "root": "#dbeafe",
        "ready": "#ffffff",
        "pending": "#fef3c7",
        "generated": "#dcfce7",
        "fallback": "#fee2e2",
        "resampled": "#e0e7ff",
        "solved": "#bbf7d0",
        "final": "#f1f5f9",
        "evaluator": "#f3e8ff",
    }
    OUTLINES = {
        "root": "#2563eb",
        "ready": "#64748b",
        "pending": "#d97706",
        "generated": "#16a34a",
        "fallback": "#dc2626",
        "resampled": "#4f46e5",
        "solved": "#15803d",
        "final": "#64748b",
        "evaluator": "#7e22ce",
    }

    def __init__(self, master):
        super().__init__(
            master,
            bg="#f8fafc",
            highlightthickness=0,
            bd=0,
            scrollregion=(0, 0, 1800, 2200),
        )
        self.nodes = {}
        self.payloads = {}
        self.row_counts = {}
        self.display_labels = {}
        self.evaluator_by_step = {}
        self.connected_edges = set()
        self.node_aliases = {}
        self.bind("<Button-1>", self.on_click)
        self.bind("<Motion>", self.on_motion)
        self.draw_background()

    def reset(self):
        self.delete("all")
        self.nodes.clear()
        self.payloads.clear()
        self.row_counts.clear()
        self.display_labels.clear()
        self.evaluator_by_step.clear()
        self.connected_edges.clear()
        self.node_aliases.clear()
        self.draw_background()

    def draw_background(self):
        self.create_rectangle(0, 0, 3000, 3000, fill="#f8fafc", outline="", tags=("background",))
        for y in range(145, 2200, 145):
            self.create_line(56, y + 42, 1320, y + 42, fill="#e2e8f0", dash=(2, 8), tags=("background",))
        self.tag_lower("background")

    def resolve_node_id(self, node_id):
        seen = set()
        while node_id in self.node_aliases and node_id not in seen:
            seen.add(node_id)
            node_id = self.node_aliases[node_id]
        return node_id

    def state_position(self, row, lane=None):
        if row < 0:
            return 570, 40
        if lane is None:
            lane = self.row_counts.get(row, 0)
            self.row_counts[row] = lane + 1
        x = 90 + lane * 155
        y = 155 + row * 145
        self.configure(scrollregion=(0, 0, max(1800, x + 360), max(2200, y + 260)))
        return x, y

    def label_for_state(self, node_id, row):
        if node_id in self.display_labels:
            return self.display_labels[node_id]
        if node_id == "root":
            label = "Root"
        else:
            match = re.search(r"(?:init-|state|pending|resampled)(\d+)$", node_id)
            state_num = int(match.group(1)) + 1 if match else self.row_counts.get(row, 1)
            label = f"Step {row}\nState {state_num}"
        self.display_labels[node_id] = label
        return label

    def remove_node(self, node_id):
        node = self.nodes.pop(node_id, None)
        self.payloads.pop(node_id, None)
        if not node:
            return
        for item in node.get("items", []):
            self.delete(item)

    def pending_id_for_state(self, node_id):
        match = re.match(r"step(\d+)-state(\d+)$", node_id)
        if not match:
            return None
        return f"step{match.group(1)}-pending{match.group(2)}"

    def add_state(self, state, row=0):
        node_id = state["id"]
        if node_id in self.nodes:
            self.update_state(state)
            return
        pending_id = self.pending_id_for_state(node_id)
        reused_center = None
        reused_label = None
        if pending_id in self.nodes:
            reused_center = self.nodes[pending_id]["center"]
            reused_label = self.display_labels.get(pending_id)
            self.remove_node(pending_id)
        if reused_center:
            x, y = reused_center[0] - 42, reused_center[1] - 42
            if reused_label:
                self.display_labels[node_id] = reused_label
        else:
            x, y = self.state_position(row)
        color = self.COLORS.get(state.get("status"), self.COLORS["ready"])
        outline = self.OUTLINES.get(state.get("status"), "#64748b")
        shadow = self.create_oval(x + 4, y + 6, x + 88, y + 90, fill="#cbd5e1", outline="", tags=("clickable", node_id))
        oval_options = {
            "fill": color,
            "outline": outline,
            "width": 2,
            "tags": ("clickable", node_id),
        }
        if state.get("status") == "pending":
            oval_options["dash"] = (5, 3)
        oval = self.create_oval(x, y, x + 84, y + 84, **oval_options)
        label = self.label_for_state(node_id, row)
        text = self.create_text(
            x + 42,
            y + 42,
            text=label,
            width=74,
            font=("Segoe UI Semibold", 9),
            fill="#0f172a",
            tags=("clickable", node_id),
        )
        self.tag_lower(shadow, oval)
        items = [shadow, oval, text]
        self.nodes[node_id] = {"shape": oval, "text": text, "items": items, "center": (x + 42, y + 42), "kind": "state", "row": row}
        self.payloads[node_id] = state
        parent_id = self.resolve_node_id(state.get("parent_id"))
        state["parent_id"] = parent_id
        if parent_id and parent_id in self.nodes:
            self.connect(parent_id, node_id)

    def update_state(self, state):
        node = self.nodes.get(state["id"])
        if not node:
            return
        self.payloads[state["id"]] = {**self.payloads.get(state["id"], {}), **state}
        color = self.COLORS.get(state.get("status"), self.COLORS["ready"])
        outline = self.OUTLINES.get(state.get("status"), "#64748b")
        self.itemconfig(node["shape"], fill=color)
        self.itemconfig(node["shape"], outline=outline)
        self.itemconfig(node["shape"], dash=(5, 3) if state.get("status") == "pending" else "")
        if state.get("status") == "solved":
            self.mark_solved(state["id"])

    def mark_solved(self, node_id):
        node = self.nodes.get(node_id)
        if not node or node.get("solved_badge"):
            return
        x, y = node["center"]
        ring = self.create_oval(
            x - 49,
            y - 49,
            x + 49,
            y + 49,
            outline="#15803d",
            width=3,
            tags=("clickable", node_id),
        )
        badge = self.create_rounded_rect(
            x - 33,
            y + 37,
            x + 33,
            y + 59,
            8,
            fill="#15803d",
            outline="",
            tags=("clickable", node_id),
        )
        badge_text = self.create_text(
            x,
            y + 48,
            text="SOLVED",
            font=("Segoe UI Semibold", 7),
            fill="#ffffff",
            tags=("clickable", node_id),
        )
        node["items"].extend([ring, badge, badge_text])
        node["solved_badge"] = True

    def add_pending(self, pending_id, parent_id, row, agent_type):
        parent_id = self.resolve_node_id(parent_id)
        self.add_state({
            "id": pending_id,
            "parent_id": parent_id,
            "status": "pending",
            "current_state": f"Model call in progress via {agent_type}",
            "serialized": {"agent_type": agent_type},
            "value": None,
        }, row)

    def add_resample_alias(self, resampled_state, source_id):
        resampled_id = resampled_state.get("id")
        if not resampled_id:
            return
        self.node_aliases[resampled_id] = self.resolve_node_id(source_id)

    def evaluator_id_for_row(self, row):
        return f"evaluator-step-{row}"

    def add_evaluator(self, evaluator_id, state_id, row):
        row_evaluator_id = self.evaluator_id_for_row(row)
        if row_evaluator_id in self.nodes:
            self.payloads[row_evaluator_id]["state_ids"].add(state_id)
            if state_id in self.nodes:
                self.connect(state_id, row_evaluator_id)
            return row_evaluator_id
        state_node = self.nodes.get(state_id)
        x = 1120
        y = (state_node["center"][1] - 28) if state_node else (155 + row * 145)
        shadow = self.create_rounded_rect(x + 4, y + 5, x + 136, y + 61, 14, fill="#cbd5e1", outline="", tags=("clickable", row_evaluator_id))
        rect = self.create_rounded_rect(
            x,
            y,
            x + 132,
            y + 56,
            14,
            fill=self.COLORS["evaluator"],
            outline=self.OUTLINES["evaluator"],
            width=2,
            tags=("clickable", row_evaluator_id),
        )
        text = self.create_text(
            x + 66,
            y + 28,
            text=f"Step {row}\nEvaluator",
            width=120,
            font=("Segoe UI Semibold", 9),
            fill="#3b0764",
            tags=("clickable", row_evaluator_id),
        )
        self.tag_lower(shadow, rect)
        self.nodes[row_evaluator_id] = {"shape": rect, "text": text, "items": [shadow, rect, text], "center": (x + 66, y + 28), "kind": "evaluator", "row": row}
        self.payloads[row_evaluator_id] = {
            "id": row_evaluator_id,
            "source_evaluator_ids": {evaluator_id},
            "state_ids": {state_id},
            "scores": {},
        }
        self.evaluator_by_step[row] = row_evaluator_id
        if state_id in self.nodes:
            self.connect(state_id, row_evaluator_id)
        return row_evaluator_id

    def complete_evaluator(self, evaluator_id, state_id, score, solved, terminal, row):
        row_evaluator_id = self.add_evaluator(evaluator_id, state_id, row)
        payload = self.payloads[row_evaluator_id]
        payload["source_evaluator_ids"].add(evaluator_id)
        payload["state_ids"].add(state_id)
        payload["scores"][state_id] = {"score": score, "solved": solved, "terminal": terminal}
        node = self.nodes.get(row_evaluator_id)
        if node:
            count = len(payload["scores"])
            self.itemconfig(node["text"], text=f"Step {row}\n{count} scored")
        if state_id in self.payloads:
            status = "solved" if solved else "generated"
            self.update_state({"id": state_id, "value": score, "status": status})

    def connect(self, source_id, target_id):
        edge = (source_id, target_id)
        if edge in self.connected_edges:
            return
        self.connected_edges.add(edge)
        sx, sy = self.nodes[source_id]["center"]
        tx, ty = self.nodes[target_id]["center"]
        midpoint_y = (sy + ty) / 2
        line = self.create_line(
            sx,
            sy + 42,
            sx,
            midpoint_y,
            tx,
            midpoint_y,
            tx,
            ty - 42,
            arrow=tk.LAST,
            fill="#94a3b8",
            width=2,
            smooth=True,
            splinesteps=18,
        )
        self.tag_lower(line, self.nodes[source_id]["shape"])

    def create_rounded_rect(self, x1, y1, x2, y2, radius, **kwargs):
        points = [
            x1 + radius, y1,
            x2 - radius, y1,
            x2, y1,
            x2, y1 + radius,
            x2, y2 - radius,
            x2, y2,
            x2 - radius, y2,
            x1 + radius, y2,
            x1, y2,
            x1, y2 - radius,
            x1, y1 + radius,
            x1, y1,
        ]
        return self.create_polygon(points, smooth=True, splinesteps=16, **kwargs)

    def on_click(self, event):
        clicked = self.find_withtag(tk.CURRENT)
        if not clicked:
            return
        tags = self.gettags(clicked[0])
        node_id = next((tag for tag in tags if tag in self.payloads), None)
        if not node_id:
            return
        show_payload_dialog(self, node_id, self.payloads[node_id])

    def on_motion(self, event):
        current = self.find_withtag(tk.CURRENT)
        if current and "clickable" in self.gettags(current[0]):
            self.configure(cursor="hand2")
        else:
            self.configure(cursor="")


def show_payload_dialog(parent, title, payload):
    dialog = tk.Toplevel(parent)
    dialog.title(title)
    dialog.geometry("760x520")
    dialog.configure(bg="#f8fafc")
    header = tk.Frame(dialog, bg="#f8fafc")
    header.pack(fill=tk.X, padx=18, pady=(16, 8))
    tk.Label(
        header,
        text=title,
        bg="#f8fafc",
        fg="#0f172a",
        font=("Segoe UI Semibold", 13),
        anchor="w",
    ).pack(fill=tk.X)
    container = tk.Frame(dialog, bg="#ffffff", highlightthickness=1, highlightbackground="#e2e8f0")
    container.pack(fill=tk.BOTH, expand=True, padx=18, pady=(0, 18))
    text = scrolledtext.ScrolledText(
        container,
        wrap=tk.WORD,
        font=("Segoe UI", 10),
        bg="#ffffff",
        fg="#0f172a",
        insertbackground="#0f172a",
        relief=tk.FLAT,
        borderwidth=0,
        padx=14,
        pady=14,
    )
    text.pack(fill=tk.BOTH, expand=True)
    lines = []
    if "scores" in payload:
        lines.append("Scores")
        lines.append("")
        scores = payload.get("scores", {})
        if scores:
            for state_id, result in sorted(scores.items()):
                label = parent.display_labels.get(state_id, state_id)
                label = " ".join(label.splitlines())
                lines.append(
                    f"{label}: score={result.get('score'):.3f}, "
                    f"solved={result.get('solved')}, terminal={result.get('terminal')}"
                )
        else:
            lines.append("No scores completed yet.")
    else:
        content = str(payload.get("current_state") or "")
        if is_markdown(content):
            configure_markdown_tags(text)
            render_markdown(text, content)
            metadata = format_metadata(payload)
            if metadata:
                text.insert(tk.END, "\n\nState metadata\n", "heading2")
                text.insert(tk.END, metadata, "metadata")
            text.configure(state=tk.DISABLED)
            return
        lines.append(content)
        lines.append("")
        lines.append("State metadata")
        lines.append(format_metadata(payload))
    text.insert("1.0", "\n".join(lines))
    text.configure(state=tk.DISABLED)


def format_metadata(payload):
    metadata = payload.get("serialized") or payload
    return json.dumps(to_jsonable(metadata), indent=2, sort_keys=True, ensure_ascii=True)


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(to_jsonable(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def is_markdown(content):
    markdown_patterns = [
        r"(?m)^#{1,6}\s+\S",
        r"(?m)^[-*+]\s+\S",
        r"(?m)^\d+\.\s+\S",
        r"(?m)^>\s+\S",
        r"```",
        r"`[^`]+`",
        r"\*\*[^*]+\*\*",
        r"__[^_]+__",
        r"(?m)^\|.+\|$",
    ]
    return any(re.search(pattern, content) for pattern in markdown_patterns)


def configure_markdown_tags(text):
    text.tag_configure("heading1", font=("Segoe UI", 16, "bold"), spacing1=8, spacing3=6)
    text.tag_configure("heading2", font=("Segoe UI", 13, "bold"), spacing1=7, spacing3=5)
    text.tag_configure("heading3", font=("Segoe UI", 11, "bold"), spacing1=6, spacing3=4)
    text.tag_configure("bold", font=("Segoe UI", 10, "bold"))
    text.tag_configure("italic", font=("Segoe UI", 10, "italic"))
    text.tag_configure("code", font=("Consolas", 10), background="#eef2ff", foreground="#312e81")
    text.tag_configure("codeblock", font=("Consolas", 10), background="#f1f5f9", lmargin1=12, lmargin2=12, spacing1=4, spacing3=4)
    text.tag_configure("bullet", lmargin1=20, lmargin2=34)
    text.tag_configure("quote", foreground="#475569", lmargin1=18, lmargin2=18)
    text.tag_configure("metadata", font=("Consolas", 9), foreground="#64748b")


def render_markdown(text, content):
    in_code_block = False
    for raw_line in content.splitlines():
        line = raw_line.rstrip()
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            text.insert(tk.END, line + "\n", "codeblock")
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            level = min(len(heading.group(1)), 3)
            insert_inline_markdown(text, heading.group(2) + "\n", f"heading{level}")
            continue

        bullet = re.match(r"^([-*+])\s+(.*)$", line)
        if bullet:
            text.insert(tk.END, "* ", "bullet")
            insert_inline_markdown(text, bullet.group(2) + "\n", "bullet")
            continue

        numbered = re.match(r"^(\d+\.)\s+(.*)$", line)
        if numbered:
            text.insert(tk.END, f"{numbered.group(1)} ", "bullet")
            insert_inline_markdown(text, numbered.group(2) + "\n", "bullet")
            continue

        quote = re.match(r"^>\s?(.*)$", line)
        if quote:
            insert_inline_markdown(text, quote.group(1) + "\n", "quote")
            continue

        insert_inline_markdown(text, line + "\n")


def insert_inline_markdown(text, line, base_tag=None):
    token_pattern = re.compile(r"(`[^`]+`|\*\*[^*]+\*\*|__[^_]+__|\*[^*\n]+\*|_[^_\n]+_)")
    pos = 0
    for match in token_pattern.finditer(line):
        if match.start() > pos:
            insert_with_optional_tag(text, line[pos:match.start()], base_tag)
        token = match.group(0)
        if token.startswith("`"):
            insert_with_optional_tag(text, token[1:-1], "code")
        elif token.startswith(("**", "__")):
            insert_with_optional_tag(text, token[2:-2], "bold")
        else:
            insert_with_optional_tag(text, token[1:-1], "italic")
        pos = match.end()
    if pos < len(line):
        insert_with_optional_tag(text, line[pos:], base_tag)


def insert_with_optional_tag(text, value, tag):
    if tag:
        text.insert(tk.END, value, tag)
    else:
        text.insert(tk.END, value)


class RunnerApp:
    def __init__(self, root, args):
        self.root = root
        self.args = args
        self.events = queue.Queue()
        self.configure_style()
        root.title("BenchmarkRW ReAgents Runner")
        root.geometry("1240x820")
        root.configure(bg="#f8fafc")

        shell = ttk.Frame(root, style="App.TFrame")
        shell.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(shell, style="App.TFrame")
        header.pack(fill=tk.X, padx=18, pady=(16, 10))
        ttk.Label(header, text="ReAgents Runner", style="Title.TLabel").pack(anchor="w")
        # ttk.Label(
        #     header,
        #     text=f"{RUNNER_BENCHMARK} / {RUNNER_SPLIT} with {RUNNER_METHOD}",
        #     style="Subtitle.TLabel",
        # ).pack(anchor="w", pady=(2, 0))

        top = ttk.Frame(shell, style="Panel.TFrame")
        top.pack(fill=tk.X, padx=18, pady=(0, 12))
        input_frame = tk.Frame(top, bg="#ffffff", highlightthickness=1, highlightbackground="#cbd5e1")
        input_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(12, 8), pady=12)
        self.question = tk.Text(
            input_frame,
            height=4,
            font=("Segoe UI", 10),
            bg="#ffffff",
            fg="#0f172a",
            insertbackground="#0f172a",
            relief=tk.FLAT,
            borderwidth=0,
            padx=12,
            pady=10,
            wrap=tk.WORD,
        )
        self.question.pack(fill=tk.X, expand=True)
        self.submit = ttk.Button(top, text="Run", command=self.start_run, width=14, style="Accent.TButton")
        self.submit.pack(side=tk.LEFT, padx=(4, 12), pady=12, ipady=16)

        body = ttk.Frame(shell, style="CanvasPanel.TFrame")
        body.pack(fill=tk.BOTH, expand=True, padx=18, pady=(0, 10))
        self.canvas = ReAgentsCanvas(body)
        xscroll = ttk.Scrollbar(shell, orient=tk.HORIZONTAL, command=self.canvas.xview)
        yscroll = ttk.Scrollbar(body, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(1, 0), pady=1)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y, pady=1)
        xscroll.pack(fill=tk.X, padx=18)

        self.status = tk.StringVar(value="Ready.")
        ttk.Label(shell, textvariable=self.status, anchor="w", style="Status.TLabel").pack(fill=tk.X, padx=18, pady=(0, 10))
        self.root.after(100, self.process_events)

    def configure_style(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("App.TFrame", background="#f8fafc")
        style.configure("Panel.TFrame", background="#ffffff", relief="flat")
        style.configure("CanvasPanel.TFrame", background="#e2e8f0", relief="flat")
        style.configure("Title.TLabel", background="#f8fafc", foreground="#0f172a", font=("Segoe UI Semibold", 16))
        style.configure("Subtitle.TLabel", background="#f8fafc", foreground="#64748b", font=("Segoe UI", 10))
        style.configure("Status.TLabel", background="#f8fafc", foreground="#475569", font=("Segoe UI", 9))
        style.configure(
            "Accent.TButton",
            background="#2563eb",
            foreground="#ffffff",
            borderwidth=0,
            focusthickness=0,
            focuscolor="#2563eb",
            font=("Segoe UI Semibold", 10),
            padding=(14, 10),
        )
        style.map(
            "Accent.TButton",
            background=[("disabled", "#93c5fd"), ("active", "#1d4ed8")],
            foreground=[("disabled", "#eff6ff"), ("active", "#ffffff")],
        )

    def start_run(self):
        question = self.question.get("1.0", tk.END).strip()
        if not question:
            messagebox.showwarning("Missing question", "Enter a question before starting.")
            return
        self.canvas.reset()
        self.submit.configure(state=tk.DISABLED)
        self.status.set(f"Running {RUNNER_METHOD} on {RUNNER_BENCHMARK}/{RUNNER_SPLIT}...")
        thread = threading.Thread(target=lambda: asyncio.run(run_reagents(question, self.args, self.events)), daemon=True)
        thread.start()

    def process_events(self):
        try:
            while True:
                event = self.events.get_nowait()
                self.handle_event(event)
        except queue.Empty:
            pass
        self.root.after(100, self.process_events)

    def handle_event(self, event):
        event_type = event.get("type")
        step = int(event.get("step", 0))
        if event_type == "run_started":
            self.canvas.add_state(event["root"], -1)
            for state in event["states"]:
                self.canvas.add_state(state, 0)
        elif event_type == "model_call_started":
            self.canvas.add_pending(event["pending_id"], event["parent_id"], step + 1, event.get("agent_type", "agent"))
        elif event_type == "state_created":
            self.canvas.add_state(event["state"], step + 1)
        elif event_type == "state_resampled":
            self.canvas.add_resample_alias(event["state"], event.get("source_id"))
        elif event_type == "evaluation_started":
            self.canvas.add_evaluator(event["evaluator_id"], event["state_id"], step + 1)
        elif event_type == "evaluation_completed":
            self.canvas.complete_evaluator(
                event["evaluator_id"],
                event["state_id"],
                float(event["score"]),
                bool(event["solved"]),
                bool(event["terminal"]),
                step + 1,
            )
        elif event_type == "run_completed":
            self.submit.configure(state=tk.NORMAL)
            final = event["states"][0] if event.get("states") else {}
            for state in event.get("states", []):
                if event.get("solved") and state.get("id") in self.canvas.nodes:
                    self.canvas.update_state({"id": state["id"], "status": "solved"})
            self.status.set("Solved." if event.get("solved") else "Finished without a jury-solved state.")
            messagebox.showinfo("Final result", final.get("current_state") or str(final.get("serialized") or final))
        elif event_type == "error":
            self.submit.configure(state=tk.NORMAL)
            self.status.set("Error.")
            messagebox.showerror("Runner error", event.get("message", "Unknown error"))


async def run_reagents(question, args, events):
    try:
        os.makedirs(os.path.dirname(args.cache_path), exist_ok=True)
        cache = Cache(args.cache_path)
        model_config = OmegaConf.load(args.model_config_path) if os.path.exists(args.model_config_path) else None

        provider = model_config["models"]["primary"]["provider"] if model_config is not None else args.provider
        api_key = model_config["models"]["primary"]["api_key"] if model_config is not None else args.api_key
        model_name = args.model
        if model_config is not None and not model_name:
            model_name = model_config["models"]["primary"]["model"]

        model = OnlineLLM(provider=provider, api_key=api_key, reasoning_effort=args.reasoning_effort)
        pipeline = OnlineAPI(model=model, cache=cache, batch_size=args.batch_size, timeout=args.timeout, allow_batch_overflow=args.allow_batch_overflow, correctness=bool(args.correctness))
        api = API(
            pipeline=pipeline,
            model=model_name,
            log_path=f"logs/raw_calls/{RUNNER_NAME}/{model_name}/{RUNNER_BENCHMARK}/{RUNNER_METHOD}_{RUNNER_SPLIT}_{RUNNER_NAME}.log",
        )
        params = DecodingParameters(args.max_completion_tokens, args.temperature, args.top_p, args.stop, args.logprobs)
        config = OmegaConf.load(f"scripts/configs/{RUNNER_BENCHMARK}.yaml")[RUNNER_METHOD]

        environment = EnvironmentFactory.get(RUNNER_BENCHMARK)
        if model_config is not None and model_config["models"]["jury"] is not None:
            environment.add_jury_evaluation(model_config["models"]["jury"])

        method = MethodFactory.get(
            method=RUNNER_METHOD,
            benchmark=RUNNER_BENCHMARK,
            params=params,
            model=api,
            env=environment,
            config=config,
        )
        method.visualization_callback = events.put
        benchmark = BenchmarkBenchmarkRW(split=RUNNER_SPLIT, question=question)
        await method.benchmark(benchmark=benchmark, ns_ratio=0.0, value_cache=True)
    except Exception as exc:
        events.put({"type": "error", "message": str(exc)})


def parse_args():
    parser = argparse.ArgumentParser(
        f"Run {RUNNER_BENCHMARK}/{RUNNER_SPLIT} with {RUNNER_METHOD} and a Tkinter visualization."
    )
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--api_key", default="OPENAI_API_KEY_CLAN")
    parser.add_argument("--model", default="gpt-4.1-nano")
    parser.add_argument("--model_config_path", default="models_config.yaml")
    parser.add_argument("--reasoning_effort", default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_completion_tokens", type=int, default=2500)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--stop", nargs="+", default=None)
    parser.add_argument("--logprobs", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--allow_batch_overflow", type=int, default=1)
    parser.add_argument("--correctness", type=int, default=1)
    parser.add_argument("--cache_path", default=f"caches/{RUNNER_NAME}/{RUNNER_BENCHMARK}/{RUNNER_METHOD}_{RUNNER_SPLIT}")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    tk_root = tk.Tk()
    RunnerApp(tk_root, cli_args)
    tk_root.mainloop()
