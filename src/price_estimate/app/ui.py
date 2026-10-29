"""Gradio dashboard.

Tab 1 "Deal hunter": runs the agent framework every 5 minutes, streams the
colour-coded agent log, lists the deals found and shows the vector store in 3D.
Tab 2 "Price a product": type any description and get the ensemble's estimate
with each member's individual guess.

    python -m price_estimate.app.ui
"""

from __future__ import annotations

import logging
import queue
import threading
import time

from price_estimate.agents import base as colors

HTML_COLORS = {
    colors.BG_BLACK + colors.RED: "#dd0000",
    colors.BG_BLACK + colors.GREEN: "#00dd00",
    colors.BG_BLACK + colors.YELLOW: "#dddd00",
    colors.BG_BLACK + colors.BLUE: "#4f6bff",
    colors.BG_BLACK + colors.MAGENTA: "#aa00dd",
    colors.BG_BLACK + colors.CYAN: "#00dddd",
    colors.BG_BLACK + colors.WHITE: "#87ceeb",
    colors.BG_BLUE + colors.WHITE: "#ff7800",
}


def ansi_to_html(message: str) -> str:
    for code, color in HTML_COLORS.items():
        message = message.replace(code, f'<span style="color:{color}">')
    return message.replace(colors.RESET, "</span>")


def log_panel(lines: list[str]) -> str:
    body = "<br>".join(lines[-18:])
    return (
        '<div style="height:400px;overflow-y:auto;border:1px solid #ccc;'
        f'background:#222229;padding:10px;font-family:monospace">{body}</div>'
    )


class QueueHandler(logging.Handler):
    def __init__(self, sink: queue.Queue):
        super().__init__()
        self.sink = sink
        self.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))

    def emit(self, record):
        self.sink.put(self.format(record))


def table_for(opportunities) -> list[list]:
    return [
        [
            o.deal.product_description,
            f"${o.deal.price:.2f}",
            f"${o.estimate:.2f}",
            f"${o.discount:.2f}",
            o.deal.url,
        ]
        for o in opportunities
    ]


def build_ui(framework=None):
    import gradio as gr
    import plotly.graph_objects as go

    from price_estimate.agents.framework import DealAgentFramework

    state = {"framework": framework}

    def get_framework():
        if state["framework"] is None:
            state["framework"] = DealAgentFramework()
        return state["framework"]

    def run_with_logging(lines):
        logs: queue.Queue = queue.Queue()
        results: queue.Queue = queue.Queue()
        handler = QueueHandler(logs)
        logging.getLogger().addHandler(handler)

        def worker():
            try:
                results.put(table_for(get_framework().run()))
            except Exception as error:  # surface failures in the log panel
                logging.error(f"run failed: {error}")
                results.put(table_for(get_framework().memory.items))

        threading.Thread(target=worker, daemon=True).start()
        table = table_for(get_framework().memory.items)
        try:
            while True:
                try:
                    lines.append(ansi_to_html(logs.get_nowait()))
                    yield lines, log_panel(lines), table
                except queue.Empty:
                    try:
                        table = results.get_nowait()
                        yield lines, log_panel(lines), table
                        break
                    except queue.Empty:
                        time.sleep(0.1)
        finally:
            logging.getLogger().removeHandler(handler)  # no duplicate handlers across runs

    def vector_plot():
        figure = go.Figure()
        try:
            from price_estimate.rag import get_collection, plot_data

            _, points, point_colors = plot_data(get_collection(), max_points=800)
            figure.add_trace(
                go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode="markers",
                    marker=dict(size=2, color=point_colors, opacity=0.7),
                )
            )
        except Exception as error:
            figure.update_layout(title=f"Vector store unavailable: {error}")
        figure.update_layout(height=400, margin=dict(r=5, b=1, l=5, t=30))
        return figure

    def price_product(description):
        from price_estimate.app.api import get_pricer

        if not description or len(description.strip()) < 3:
            return "Please describe the product.", []
        pricer = get_pricer()
        estimate = pricer.price(description)
        rows = [
            [name, f"${value:.2f}", f"{pricer.weights.get(name, 0):.2f}"]
            for name, value in pricer.last_breakdown.items()
        ]
        return f"## Estimated price: ${estimate:,.2f}", rows

    def resend(evt: gr.SelectData):
        items = get_framework().memory.items
        row = evt.index[0]
        if 0 <= row < len(items):
            from price_estimate.agents.messaging import MessagingAgent

            MessagingAgent().alert(items[row])

    with gr.Blocks(title="Price Estimate", fill_width=True) as ui:
        gr.Markdown(
            "# Price Estimate\nAn agent system that prices products from text and hunts for underpriced deals."
        )
        with gr.Tab("Deal hunter"):
            lines = gr.State([])
            deals = gr.Dataframe(
                headers=["Deal", "Price", "Estimate", "Discount", "URL"],
                wrap=True,
                column_widths=[6, 1, 1, 1, 3],
                max_height=400,
            )
            with gr.Row():
                logs = gr.HTML()
                gr.Plot(value=vector_plot, show_label=False)
            deals.select(resend)
            ui.load(run_with_logging, inputs=[lines], outputs=[lines, logs, deals])
            gr.Timer(value=300).tick(run_with_logging, inputs=[lines], outputs=[lines, logs, deals])
        with gr.Tab("Price a product"):
            text = gr.Textbox(
                label="Product description",
                lines=4,
                placeholder="e.g. Sony WH-1000XM5 wireless noise cancelling headphones",
            )
            button = gr.Button("Estimate", variant="primary")
            result = gr.Markdown()
            breakdown = gr.Dataframe(headers=["Model", "Estimate", "Weight"])
            button.click(price_product, inputs=[text], outputs=[result, breakdown])
    return ui


if __name__ == "__main__":
    build_ui().launch(inbrowser=True)
