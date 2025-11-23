#!/usr/bin/env python3
"""Quick web UI for GR Cup strategy engine."""
from flask import Flask, render_template, jsonify, send_from_directory
from pathlib import Path
import json

app = Flask(__name__)
BASE_DIR = Path(__file__).parent
REPORTS_DIR = BASE_DIR / "reports"


@app.route("/")
def index():
    """Main dashboard page."""
    return render_template("index.html")


@app.route("/api/scenarios")
def list_scenarios():
    """List available scenarios."""
    scenarios = []
    if REPORTS_DIR.exists():
        for scenario_dir in sorted(REPORTS_DIR.iterdir()):
            if scenario_dir.is_dir() and (scenario_dir / "validation_report.json").exists():
                scenarios.append(scenario_dir.name)
    return jsonify(scenarios)


@app.route("/api/report/<scenario>")
def get_report(scenario):
    """Get validation report for a scenario."""
    report_file = REPORTS_DIR / scenario / "validation_report.json"
    if report_file.exists():
        with open(report_file) as f:
            return jsonify(json.load(f))
    return jsonify({"error": f"Report not found for {scenario}"}), 404


@app.route("/api/counterfactuals/<scenario>")
def get_counterfactuals(scenario):
    """Get counterfactuals for a scenario."""
    cf_file = REPORTS_DIR / scenario / "counterfactuals.json"
    if cf_file.exists():
        with open(cf_file) as f:
            return jsonify(json.load(f))
    return jsonify({"error": f"Counterfactuals not found for {scenario}"}), 404


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

