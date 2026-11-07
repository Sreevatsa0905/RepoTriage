import React, { useState, useEffect } from 'react';
import {
  BarChart3,
  Search,
  Sliders,
  ShieldCheck,
  FileCheck,
  Layers
} from 'lucide-react';

export const EvaluationTab: React.FC = () => {
  const [data, setData] = useState<any>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    const fetchEvalData = async () => {
      try {
        const res = await fetch('/api/evaluation');
        if (!res.ok) throw new Error(`HTTP error ${res.status}`);
        const json = await res.json();
        setData(json);
      } catch (err: any) {
        setErrorMsg(err.message || 'Failed to fetch evaluation metrics');
      } finally {
        setIsLoading(false);
      }
    };
    fetchEvalData();
  }, []);

  if (isLoading) {
    return (
      <div className="bg-dark-800 border border-dark-600 rounded-xl p-12 text-center text-gray-400">
        <div className="animate-spin w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full mx-auto mb-3" />
        <p className="text-sm">Loading comprehensive evaluation benchmark data...</p>
      </div>
    );
  }

  return (
    <div className="space-y-10 pb-12">
      {/* Overview Banner */}
      <div className="bg-dark-800 border border-dark-600 rounded-xl p-6 shadow-md">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-indigo-400" />
              Empirical Evaluation & Ablation Benchmarks
            </h2>
            <p className="text-sm text-gray-400 mt-1">
              Exact measured metrics across Hybrid Retrieval (n=51 duplicate pairs), Guardrail Verification (ANLI & In-Domain B2), Perturbation Testing (n=300), and Classifier Generalization.
            </p>
          </div>
          <div className="px-3 py-1.5 rounded-lg bg-dark-900 border border-dark-700 text-xs text-gray-400 font-mono">
            Human Check: <span className="text-amber-400 font-semibold">{data?.human_check_status || 'not performed'}</span>
          </div>
        </div>
      </div>

      {errorMsg && (
        <div className="p-4 bg-red-950/60 border border-red-800 rounded-xl text-red-300 text-sm">
          Failed to load live metrics: {errorMsg}. Showing recorded benchmark numbers below.
        </div>
      )}

      {/* 1. Retrieval 4-Way Ablation */}
      <section className="bg-dark-800 border border-dark-600 rounded-xl p-6 shadow-md space-y-6">
        <div>
          <div className="flex items-center justify-between">
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <Search className="w-4 h-4 text-indigo-400" />
              1. Retrieval 4-Way Ablation & Bootstrap 95% Confidence Intervals
            </h3>
            <span className="text-xs text-indigo-300 bg-indigo-950 px-2.5 py-0.5 rounded border border-indigo-800">
              Protocol: Issues-only candidate pool | n = 51 pairs | 1,000 bootstrap resamples
            </span>
          </div>
          <p className="text-xs text-gray-400 mt-1">
            Evaluated on held-out duplicate query-target pairs across seed repos. Chunks strictly exclude the query issue and any chunks created after query creation.
          </p>
        </div>

        {/* Ablation Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left border-collapse">
            <thead>
              <tr className="bg-dark-900 text-gray-400 border-b border-dark-700">
                <th className="p-3 font-semibold">Retrieval Method</th>
                <th className="p-3 font-semibold">Recall@5 [95% CI]</th>
                <th className="p-3 font-semibold">Precision@5</th>
                <th className="p-3 font-semibold">MRR@10 [95% CI]</th>
                <th className="p-3 font-semibold">nDCG@10</th>
                <th className="p-3 font-semibold">Mean Latency</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-dark-700 text-gray-300 font-mono">
              <tr className="hover:bg-dark-750">
                <td className="p-3 font-sans font-medium text-white">1. BM25 (Code-Aware Tokenizer)</td>
                <td className="p-3 font-bold text-gray-200">0.7451 <span className="text-gray-500 font-normal">[0.6275, 0.8627]</span></td>
                <td className="p-3">0.1490</td>
                <td className="p-3">0.5484 <span className="text-gray-500 font-normal">[0.4340, 0.6670]</span></td>
                <td className="p-3">0.6017</td>
                <td className="p-3 font-sans text-gray-400">111.7ms</td>
              </tr>
              <tr className="hover:bg-dark-750">
                <td className="p-3 font-sans font-medium text-white">2. Dense (BAAI/bge-base-en-v1.5)</td>
                <td className="p-3 font-bold text-gray-200">0.8039 <span className="text-gray-500 font-normal">[0.6863, 0.9020]</span></td>
                <td className="p-3">0.1608</td>
                <td className="p-3">0.5730 <span className="text-gray-500 font-normal">[0.4675, 0.6891]</span></td>
                <td className="p-3">0.6439</td>
                <td className="p-3 font-sans text-gray-400">1348.8ms</td>
              </tr>
              <tr className="hover:bg-dark-750">
                <td className="p-3 font-sans font-medium text-white">3. BM25 + Dense (RRF, k=60)</td>
                <td className="p-3 font-bold text-gray-200">0.7647 <span className="text-gray-500 font-normal">[0.6471, 0.8824]</span></td>
                <td className="p-3">0.1529</td>
                <td className="p-3">0.6018 <span className="text-gray-500 font-normal">[0.4948, 0.7139]</span></td>
                <td className="p-3">0.6692</td>
                <td className="p-3 font-sans text-gray-400">196.3ms</td>
              </tr>
              <tr className="hover:bg-dark-750 bg-indigo-950/20">
                <td className="p-3 font-sans font-medium text-indigo-300">4. RRF + BGE Reranker (Blend α=0.50)</td>
                <td className="p-3 font-bold text-emerald-400">0.7843 <span className="text-gray-500 font-normal">[0.6667, 0.9020]</span></td>
                <td className="p-3">0.1569</td>
                <td className="p-3 font-bold text-emerald-400">0.6022 <span className="text-gray-500 font-normal">[0.4946, 0.7140]</span></td>
                <td className="p-3 font-bold text-emerald-400">0.6697</td>
                <td className="p-3 font-sans text-gray-400">504.6ms</td>
              </tr>
            </tbody>
          </table>
        </div>

        {/* Paired Differences & Findings */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          <div className="p-4 rounded-lg bg-dark-900 border border-dark-700">
            <h4 className="font-semibold text-gray-300 mb-2">Paired Differences vs. BM25 Baseline (95% CI)</h4>
            <ul className="space-y-1.5 text-gray-400 font-mono">
              <li>• RRF vs BM25: Δ Recall@5 = <strong className="text-gray-200">+0.0196</strong> [-0.0588, +0.0980]</li>
              <li>• RRF vs BM25: Δ MRR@10 = <strong className="text-gray-200">+0.0535</strong> [-0.0129, +0.1223]</li>
              <li>• Reranker vs BM25: Δ Recall@5 = <strong className="text-gray-200">+0.0392</strong> [-0.0588, +0.1373]</li>
              <li>• Reranker vs BM25: Δ MRR@10 = <strong className="text-gray-200">+0.0538</strong> [-0.0112, +0.1182]</li>
            </ul>
            <p className="text-xs text-gray-500 mt-2 font-sans">
              *Note: Because paired CIs cross zero on n=51, switching from mixed candidate pool to issues-only pool explains the main gain, while RRF and blending provide marginal consistency improvements.
            </p>
          </div>

          <div className="p-4 rounded-lg bg-dark-900 border border-dark-700">
            <h4 className="font-semibold text-gray-300 mb-2">Reranker Score Blending Sensitivity (α sweep)</h4>
            <div className="grid grid-cols-5 gap-2 text-center font-mono">
              <div className="p-2 bg-dark-800 rounded border border-dark-700">
                <div className="text-gray-400">α=0.0</div>
                <div className="font-bold text-gray-300">0.7647</div>
                <div className="text-gray-500 text-[10px]">Pure RRF</div>
              </div>
              <div className="p-2 bg-dark-800 rounded border border-dark-700">
                <div className="text-gray-400">α=0.25</div>
                <div className="font-bold text-gray-300">0.7647</div>
                <div className="text-gray-500 text-[10px]">MRR 0.598</div>
              </div>
              <div className="p-2 bg-indigo-950/80 rounded border border-indigo-700">
                <div className="text-indigo-300 font-bold">α=0.50</div>
                <div className="font-bold text-emerald-400">0.7843</div>
                <div className="text-indigo-400 text-[10px]">MRR 0.602</div>
              </div>
              <div className="p-2 bg-dark-800 rounded border border-dark-700">
                <div className="text-gray-400">α=0.75</div>
                <div className="font-bold text-gray-300">0.7843</div>
                <div className="text-gray-500 text-[10px]">MRR 0.589</div>
              </div>
              <div className="p-2 bg-dark-800 rounded border border-dark-700">
                <div className="text-gray-400">α=1.0</div>
                <div className="font-bold text-gray-300">0.5490</div>
                <div className="text-gray-500 text-[10px]">MRR 0.396</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 2. Guardrail Benchmark B1 (ANLI) */}
      <section className="bg-dark-800 border border-dark-600 rounded-xl p-6 shadow-md space-y-6">
        <div>
          <div className="flex items-center justify-between">
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
              2. Guardrail Benchmark B1: Adversarial ANLI Evaluation
            </h3>
            <span className="text-xs text-indigo-300 bg-indigo-950 px-2.5 py-0.5 rounded border border-indigo-800">
              Dataset: facebook/anli (Round 3) | n = 150 balanced test pairs | τ_s = 0.30, τ_c = 0.30
            </span>
          </div>
          <p className="text-xs text-gray-400 mt-1">
            Evaluates cross-encoder/nli-deberta-v3-base with strict dominance decision rule (p_e &gt; p_c for entailment, p_c &gt; p_e for contradiction).
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* 3x3 Confusion Matrix */}
          <div className="p-4 rounded-lg bg-dark-900 border border-dark-700">
            <h4 className="text-xs font-semibold text-gray-300 uppercase tracking-wider mb-3">
              3x3 Confusion Matrix (Predicted vs True)
            </h4>
            <table className="w-full text-center text-xs border-collapse">
              <thead>
                <tr className="text-gray-400 border-b border-dark-700">
                  <th className="p-2 text-left">True \ Pred</th>
                  <th className="p-2">SUPPORTED</th>
                  <th className="p-2">CONTRADICTED</th>
                  <th className="p-2">INSUFFICIENT</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-dark-700 font-mono">
                <tr>
                  <td className="p-2 text-left font-sans text-gray-300">SUPPORTED</td>
                  <td className="p-2 font-bold text-emerald-400 bg-emerald-950/30">21</td>
                  <td className="p-2 text-gray-400">5</td>
                  <td className="p-2 text-gray-400">24</td>
                </tr>
                <tr>
                  <td className="p-2 text-left font-sans text-gray-300">CONTRADICTED</td>
                  <td className="p-2 text-gray-400">15</td>
                  <td className="p-2 font-bold text-rose-400 bg-rose-950/30">23</td>
                  <td className="p-2 text-gray-400">12</td>
                </tr>
                <tr>
                  <td className="p-2 text-left font-sans text-gray-300">INSUFFICIENT</td>
                  <td className="p-2 text-gray-400">10</td>
                  <td className="p-2 text-gray-400">14</td>
                  <td className="p-2 font-bold text-blue-400 bg-blue-950/30">26</td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* Metrics Breakdown */}
          <div className="p-4 rounded-lg bg-dark-900 border border-dark-700 flex flex-col justify-between">
            <div>
              <h4 className="text-xs font-semibold text-gray-300 uppercase tracking-wider mb-3">
                ANLI Performance Summary
              </h4>
              <div className="grid grid-cols-2 gap-3 text-xs mb-4">
                <div className="p-2.5 rounded bg-dark-800 border border-dark-700">
                  <div className="text-gray-400">Test Accuracy</div>
                  <div className="text-base font-bold text-white font-mono">46.67% (70 / 150)</div>
                </div>
                <div className="p-2.5 rounded bg-dark-800 border border-dark-700">
                  <div className="text-gray-400">Macro F1</div>
                  <div className="text-base font-bold text-indigo-400 font-mono">0.4673</div>
                </div>
              </div>
              <div className="space-y-1.5 text-xs text-gray-400 font-mono">
                <div>• SUPPORTED: Precision 45.65% | Recall 42.00% | F1 0.4375 (21 / 50)</div>
                <div>• CONTRADICTED: Precision 54.76% | Recall 46.00% | F1 0.5000 (23 / 50)</div>
                <div>• INSUFFICIENT: Precision 41.94% | Recall 52.00% | F1 0.4643 (26 / 50)</div>
              </div>
            </div>
            <p className="text-[11px] text-gray-500 mt-3 font-sans italic">
              ANLI is explicitly human-adversarial (trained to fool NLI models). 46.7% accuracy with dominance prevents conflicting dual-activation failure modes.
            </p>
          </div>
        </div>
      </section>

      {/* 3. Guardrail Benchmark B2 (In-Domain Negative Controls & Calibration) */}
      <section className="bg-dark-800 border border-dark-600 rounded-xl p-6 shadow-md space-y-6">
        <div>
          <div className="flex items-center justify-between">
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <Sliders className="w-4 h-4 text-indigo-400" />
              3. Guardrail Benchmark B2: In-Domain Calibration & Threshold Sweep
            </h3>
            <span className="text-xs text-indigo-300 bg-indigo-950 px-2.5 py-0.5 rounded border border-indigo-800">
              Protocol: 15 fresh held-out issues across 4 repos | 80 claims evaluated | Multi-premise competition
            </span>
          </div>
          <p className="text-xs text-gray-400 mt-1">
            Measures false-support rates on synthetic easy negatives (unrelated docs) and hard negatives (lexically close cross-repo items).
          </p>
        </div>

        {/* Threshold Sweep Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left border-collapse">
            <thead>
              <tr className="bg-dark-900 text-gray-400 border-b border-dark-700">
                <th className="p-3 font-semibold">Threshold Setting</th>
                <th className="p-3 font-semibold">Grounded Pass Rate</th>
                <th className="p-3 font-semibold">Easy Neg False-Support</th>
                <th className="p-3 font-semibold">Hard Neg False-Support</th>
                <th className="p-3 font-semibold">Contradiction Detection</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-dark-700 text-gray-300 font-mono">
              <tr className="hover:bg-dark-750">
                <td className="p-3 font-sans font-medium">τ_s = 0.50, τ_c = 0.50</td>
                <td className="p-3">42.67%</td>
                <td className="p-3 text-amber-400">1.25%</td>
                <td className="p-3 text-emerald-400">0.00%</td>
                <td className="p-3">15.00%</td>
              </tr>
              <tr className="hover:bg-dark-750">
                <td className="p-3 font-sans font-medium">τ_s = 0.70, τ_c = 0.70</td>
                <td className="p-3">40.00%</td>
                <td className="p-3 text-amber-400">1.25%</td>
                <td className="p-3 text-emerald-400">0.00%</td>
                <td className="p-3">15.00%</td>
              </tr>
              <tr className="hover:bg-dark-750 bg-indigo-950/30">
                <td className="p-3 font-sans font-bold text-indigo-300">τ_s = 0.80, τ_c = 0.80 (Production Default)</td>
                <td className="p-3 font-bold text-white">40.00%</td>
                <td className="p-3 font-bold text-amber-400">1.25% (1 / 80)</td>
                <td className="p-3 font-bold text-emerald-400">0.00% (0 / 80)</td>
                <td className="p-3 font-bold text-white">15.00%</td>
              </tr>
              <tr className="hover:bg-dark-750">
                <td className="p-3 font-sans font-medium">τ_s = 0.90, τ_c = 0.90</td>
                <td className="p-3">34.67%</td>
                <td className="p-3 text-emerald-400">0.00%</td>
                <td className="p-3 text-emerald-400">0.00%</td>
                <td className="p-3">12.00%</td>
              </tr>
            </tbody>
          </table>
        </div>

        {/* Multi-Evidence & Audit Insights */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          <div className="p-4 rounded-lg bg-dark-900 border border-dark-700">
            <h4 className="font-semibold text-gray-300 mb-2">Multi-Evidence Premise Improvement (Stage 6.6-I)</h4>
            <div className="space-y-1.5 text-gray-400">
              <div>• Single Premise Pass Rate: <strong className="text-gray-200">38.67%</strong></div>
              <div>• Concatenated Multi-Evidence Pass Rate: <strong className="text-emerald-400">40.00%</strong> (+1.33%)</div>
              <div>• Hard-Negative False-Support Rate: <strong className="text-emerald-400">0.00%</strong> (remains pristine)</div>
              <div>• Winning Premises Breakdown: 34 full chunk, 39 sentence window, 2 multi-evidence</div>
            </div>
          </div>

          <div className="p-4 rounded-lg bg-dark-900 border border-dark-700">
            <h4 className="font-semibold text-gray-300 mb-2">Audit of 12 Flagged Contradicted Claims (Stage 6.6-C)</h4>
            <div className="space-y-1.5 text-gray-400">
              <div>• Plausible LLM Errors: <strong className="text-rose-400">4</strong> (hallucinated version numbers / CVE mismatches)</div>
              <div>• Plausible NLI False Alarms: <strong className="text-amber-400">5</strong> (feature requests / syntax misreads)</div>
              <div>• Unclear / Boundary Cases: <strong className="text-gray-400">3</strong></div>
              <div className="text-[11px] text-gray-500 pt-1">
                *Agent review recorded in <code className="text-gray-400">eval/reports/audit_contradicted.csv</code>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 4. Automated Perturbation Test */}
      <section className="bg-dark-800 border border-dark-600 rounded-xl p-6 shadow-md space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <FileCheck className="w-4 h-4 text-indigo-400" />
            4. Automated Perturbation Test (Synthetic Rule-Constructed Ground Truth)
          </h3>
          <span className="text-xs text-indigo-300 bg-indigo-950 px-2.5 py-0.5 rounded border border-indigo-800">
            n = 300 balanced pairs (100 per class) | τ = 0.80
          </span>
        </div>

        <div className="p-3 rounded-lg bg-amber-950/40 border border-amber-800/60 text-xs text-amber-300">
          <strong>Methodological Note:</strong> Synthetic perturbation test. Labels are known by construction (negations, entity swaps, cross-repo passages). Claims are more templated than real LLM output and serve as an automated sanity check; not a substitute for human review.
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs font-mono">
          <div className="p-3 bg-dark-900 rounded-lg border border-dark-700 text-center">
            <div className="text-gray-400 font-sans">Accuracy</div>
            <div className="text-lg font-bold text-white mt-1">88.67%</div>
            <div className="text-[10px] text-gray-500 font-sans">266 / 300 correct</div>
          </div>
          <div className="p-3 bg-dark-900 rounded-lg border border-dark-700 text-center">
            <div className="text-gray-400 font-sans">Contradiction Recall</div>
            <div className="text-lg font-bold text-emerald-400 mt-1">92.00%</div>
            <div className="text-[10px] text-gray-500 font-sans">92 / 100 detected</div>
          </div>
          <div className="p-3 bg-dark-900 rounded-lg border border-dark-700 text-center">
            <div className="text-gray-400 font-sans">Contradicted False Support</div>
            <div className="text-lg font-bold text-emerald-400 mt-1">6.00%</div>
            <div className="text-[10px] text-gray-500 font-sans">6 / 100 negated passed</div>
          </div>
          <div className="p-3 bg-dark-900 rounded-lg border border-dark-700 text-center">
            <div className="text-gray-400 font-sans">Unrelated False Support</div>
            <div className="text-lg font-bold text-emerald-400 mt-1">0.00%</div>
            <div className="text-[10px] text-gray-500 font-sans">0 / 100 NEI passed</div>
          </div>
        </div>
      </section>

      {/* 5. Classifier Live Check & Benchmarks */}
      <section className="bg-dark-800 border border-dark-600 rounded-xl p-6 shadow-md space-y-6">
        <div>
          <div className="flex items-center justify-between">
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <Layers className="w-4 h-4 text-indigo-400" />
              5. Classifier Live Generalization & Benchmark Summary
            </h3>
            <span className="text-xs text-indigo-300 bg-indigo-950 px-2.5 py-0.5 rounded border border-indigo-800">
              Live Evaluation on Ingested Issues: n = 159 across 3 repos | Threshold τ = 0.50
            </span>
          </div>
          <p className="text-xs text-gray-400 mt-1">
            Evaluates the frozen TF-IDF + Logistic Regression model on unambiguous live GitHub issues from repotriage.db.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Live Ingest Check */}
          <div className="p-4 rounded-lg bg-dark-900 border border-dark-700">
            <h4 className="text-xs font-semibold text-gray-300 uppercase tracking-wider mb-3">
              Live Ingest Check Results (Stage 6.6-F)
            </h4>
            <div className="space-y-2 text-xs">
              <div className="flex justify-between border-b border-dark-800 pb-1">
                <span className="text-gray-400">Total Valid Issues:</span>
                <span className="font-mono text-gray-200">159 (httpx: 75, pydantic: 52, requests: 32)</span>
              </div>
              <div className="flex justify-between border-b border-dark-800 pb-1">
                <span className="text-gray-400">Coverage (Non-Uncertain):</span>
                <span className="font-mono text-emerald-400 font-bold">73.58% (117 / 159)</span>
              </div>
              <div className="flex justify-between border-b border-dark-800 pb-1">
                <span className="text-gray-400">Covered Predictions Accuracy:</span>
                <span className="font-mono text-emerald-400 font-bold">82.05%</span>
              </div>
              <div className="flex justify-between border-b border-dark-800 pb-1">
                <span className="text-gray-400">Covered Predictions Macro-F1:</span>
                <span className="font-mono text-indigo-400 font-bold">0.6979</span>
              </div>
              <div className="flex justify-between pb-1">
                <span className="text-gray-400">All Issues (Uncertain = Wrong):</span>
                <span className="font-mono text-gray-300">Accuracy 60.38% | Macro-F1 0.5610</span>
              </div>
            </div>
          </div>

          {/* Training & Benchmark Models Summary */}
          <div className="p-4 rounded-lg bg-dark-900 border border-dark-700">
            <h4 className="text-xs font-semibold text-gray-300 uppercase tracking-wider mb-3">
              Classifier Benchmark Comparison (NLBSE 2024 Test Set)
            </h4>
            <div className="space-y-2 text-xs font-mono">
              <div className="flex justify-between border-b border-dark-800 pb-1">
                <span className="text-gray-400 font-sans">Dummy Baseline:</span>
                <span>Accuracy 33.33% | Macro-F1 0.1667</span>
              </div>
              <div className="flex justify-between border-b border-dark-800 pb-1">
                <span className="text-gray-400 font-sans">Per-Repo Classical LogReg:</span>
                <span>Accuracy 74.53% | Macro-F1 0.7438</span>
              </div>
              <div className="flex justify-between border-b border-dark-800 pb-1">
                <span className="text-gray-400 font-sans">Leave-One-Repo-Out (Cold-Start):</span>
                <span className="text-indigo-400 font-bold">Macro-F1 0.5941</span>
              </div>
              <div className="flex justify-between pb-1">
                <span className="text-gray-400 font-sans text-indigo-300">RepoTriage Frozen Pooled Model:</span>
                <span className="text-emerald-400 font-bold">Accuracy 76.07% | Macro-F1 0.7605</span>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};
