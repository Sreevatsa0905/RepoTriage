import React, { useState, useEffect } from 'react';
import {
  Search,
  Sparkles,
  ExternalLink,
  ShieldCheck,
  AlertTriangle,
  XCircle,
  HelpCircle,
  ChevronDown,
  ChevronUp,
  Clock,
  CheckCircle,
  Layers,
  BookOpen
} from 'lucide-react';
import type { AnalyzeResponse, SearchResponse, Repository, EvidenceItem } from '../types';

interface IssueCopilotProps {
  repositories: Repository[];
}

export const IssueCopilot: React.FC<IssueCopilotProps> = ({ repositories }) => {
  const [selectedRepo, setSelectedRepo] = useState<string>('pydantic/pydantic');
  const [inputText, setInputText] = useState<string>('');
  const [issueUrl, setIssueUrl] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [loadingStep, setLoadingStep] = useState<number>(0);
  const [searchOnlyResults, setSearchOnlyResults] = useState<EvidenceItem[] | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalyzeResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [expandedClaims, setExpandedClaims] = useState<Record<string, boolean>>({});
  const [expandedEvidence, setExpandedEvidence] = useState<Record<string, boolean>>({});

  useEffect(() => {
    if (repositories.length > 0 && !selectedRepo) {
      setSelectedRepo(repositories[0].repo);
    }
  }, [repositories, selectedRepo]);

  // Stepper timer effect when analyzing
  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (isLoading) {
      setLoadingStep(1);
      const start = Date.now();
      interval = setInterval(() => {
        const elapsed = (Date.now() - start) / 1000;
        if (elapsed < 1.0) setLoadingStep(1);
        else if (elapsed < 2.5) setLoadingStep(2);
        else if (elapsed < 25.0) setLoadingStep(3);
        else setLoadingStep(4);
      }, 500);
    } else {
      setLoadingStep(0);
    }
    return () => clearInterval(interval);
  }, [isLoading]);

  const handleAnalyze = async () => {
    if (!inputText.trim() && !issueUrl.trim()) {
      setErrorMsg('Please enter an issue description or paste a GitHub issue URL.');
      return;
    }

    setIsLoading(true);
    setErrorMsg(null);
    setSearchOnlyResults(null);

    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: inputText,
          repo: selectedRepo,
          issue_url: issueUrl || null,
        }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Analysis failed' }));
        throw new Error(err.detail || `Server error: ${res.status}`);
      }

      const data: AnalyzeResponse = await res.json();
      setAnalysisResult(data);
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to complete analysis');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSearchOnly = async () => {
    if (!inputText.trim()) {
      setErrorMsg('Please enter a query in the text area to search.');
      return;
    }

    setIsLoading(true);
    setErrorMsg(null);
    setAnalysisResult(null);

    try {
      const res = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: inputText,
          repo: selectedRepo,
          top_k: 6,
          method: 'hybrid_rerank',
        }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Search failed' }));
        throw new Error(err.detail || `Server error: ${res.status}`);
      }

      const data: SearchResponse = await res.json();
      setSearchOnlyResults(data.results);
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to execute hybrid search');
    } finally {
      setIsLoading(false);
    }
  };

  const toggleClaimExpansion = (id: string) => {
    setExpandedClaims((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const toggleEvidenceExpansion = (id: string) => {
    setExpandedEvidence((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  return (
    <div className="space-y-8">
      {/* Input Section */}
      <div className="bg-dark-800 border border-dark-600 rounded-xl p-6 shadow-lg">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-4">
          <div>
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-indigo-400" />
              Issue Triage & Resolution Copilot
            </h2>
            <p className="text-sm text-gray-400 mt-1">
              Select target repo, input issue text or URL, and trigger hybrid retrieval with NLI-verified synthesis.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <label className="text-xs font-medium text-gray-400 uppercase tracking-wider">Repository:</label>
            <select
              value={selectedRepo}
              onChange={(e) => setSelectedRepo(e.target.value)}
              className="bg-dark-900 border border-dark-600 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-indigo-500"
            >
              {repositories.map((r) => (
                <option key={r.repo} value={r.repo}>
                  {r.repo} ({r.total_chunks} chunks)
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-xs font-medium text-gray-300 mb-1.5">
              Issue Text (Title & Body)
            </label>
            <textarea
              rows={5}
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="e.g. Field validator mode='after' fails with NoneType error when field is Optional[str] = None..."
              className="w-full bg-dark-900 border border-dark-600 rounded-lg p-3 text-sm text-white focus:outline-none focus:border-indigo-500 placeholder-gray-500 resize-y"
            />
          </div>

          <div>
            <label className="block text-xs font-medium text-gray-400 mb-1">
              Or GitHub Issue URL (Optional)
            </label>
            <input
              type="text"
              value={issueUrl}
              onChange={(e) => setIssueUrl(e.target.value)}
              placeholder="https://github.com/pydantic/pydantic/issues/1234"
              className="w-full bg-dark-900 border border-dark-600 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-indigo-500 placeholder-gray-500"
            />
          </div>

          {errorMsg && (
            <div className="p-3 rounded-lg bg-red-950/60 border border-red-800 text-red-300 text-sm flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 flex-shrink-0" />
              <span>{errorMsg}</span>
            </div>
          )}

          <div className="flex items-center justify-between pt-2">
            <span className="text-xs text-gray-500">
              Pipeline uses Qwen 3.5 4B (Ollama) + DeBERTa-v3 NLI verification on RTX 3050.
            </span>
            <div className="flex items-center gap-3">
              <button
                onClick={handleSearchOnly}
                disabled={isLoading}
                className="px-4 py-2 rounded-lg bg-dark-700 hover:bg-dark-600 text-gray-200 text-sm font-medium border border-dark-600 transition-colors flex items-center gap-2 disabled:opacity-50"
              >
                <Search className="w-4 h-4" />
                Search Chunks Only
              </button>
              <button
                onClick={handleAnalyze}
                disabled={isLoading}
                className="px-5 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-semibold shadow-md shadow-indigo-600/30 transition-colors flex items-center gap-2 disabled:opacity-50"
              >
                <Sparkles className="w-4 h-4" />
                {isLoading ? 'Analyzing...' : 'Analyze Issue'}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Loading Progress State */}
      {isLoading && (
        <div className="bg-dark-800 border border-indigo-900/50 rounded-xl p-6 shadow-xl animate-pulse">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-base font-semibold text-white flex items-center gap-2">
              <Clock className="w-5 h-5 text-indigo-400 animate-spin" />
              Executing Multi-Stage Triage Pipeline
            </h3>
            <span className="text-xs text-indigo-300 bg-indigo-950/80 px-2.5 py-1 rounded-full border border-indigo-800">
              Estimated duration: ~25-35s (local hardware)
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-3 text-xs">
            <div className={`p-3 rounded-lg border ${loadingStep >= 1 ? 'bg-dark-700 border-indigo-500 text-white' : 'bg-dark-900 border-dark-700 text-gray-500'}`}>
              <div className="font-semibold mb-1 flex items-center gap-1.5">
                {loadingStep > 1 ? <CheckCircle className="w-3.5 h-3.5 text-emerald-400" /> : <div className="w-2 h-2 rounded-full bg-indigo-400" />}
                1. Classification
              </div>
              <p className="text-gray-400">TF-IDF LogReg cold-start model</p>
            </div>

            <div className={`p-3 rounded-lg border ${loadingStep >= 2 ? 'bg-dark-700 border-indigo-500 text-white' : 'bg-dark-900 border-dark-700 text-gray-500'}`}>
              <div className="font-semibold mb-1 flex items-center gap-1.5">
                {loadingStep > 2 ? <CheckCircle className="w-3.5 h-3.5 text-emerald-400" /> : <div className="w-2 h-2 rounded-full bg-indigo-400" />}
                2. Hybrid Retrieval
              </div>
              <p className="text-gray-400">BM25 + BGE Dense + Reranker</p>
            </div>

            <div className={`p-3 rounded-lg border ${loadingStep >= 3 ? 'bg-dark-700 border-indigo-500 text-white' : 'bg-dark-900 border-dark-700 text-gray-500'}`}>
              <div className="font-semibold mb-1 flex items-center gap-1.5">
                {loadingStep > 3 ? <CheckCircle className="w-3.5 h-3.5 text-emerald-400" /> : <div className="w-2 h-2 rounded-full bg-indigo-400" />}
                3. LLM Synthesis
              </div>
              <p className="text-gray-400">Qwen 3.5 4B structured triage</p>
            </div>

            <div className={`p-3 rounded-lg border ${loadingStep >= 4 ? 'bg-dark-700 border-indigo-500 text-white' : 'bg-dark-900 border-dark-700 text-gray-500'}`}>
              <div className="font-semibold mb-1 flex items-center gap-1.5">
                <div className="w-2 h-2 rounded-full bg-indigo-400" />
                4. NLI Verification
              </div>
              <p className="text-gray-400">DeBERTa-v3 multi-premise checks</p>
            </div>
          </div>
        </div>
      )}

      {/* Analysis Results View */}
      {analysisResult && (
        <div className="space-y-6">
          {/* Issue Type Side-by-Side Panel */}
          <div className="bg-dark-800 border border-dark-600 rounded-xl p-5 shadow-md">
            <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-4 flex items-center gap-2">
              <Layers className="w-4 h-4 text-indigo-400" />
              Issue Classification (Classifier vs. LLM)
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Frozen Classifier */}
              <div className="p-4 rounded-lg bg-dark-900 border border-dark-700">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs text-gray-400 font-medium">Frozen Classifier (TF-IDF + LogReg)</span>
                  <span className={`px-2 py-0.5 rounded text-xs font-semibold uppercase ${
                    analysisResult.classifier.predicted_label === 'bug'
                      ? 'bg-rose-950 text-rose-300 border border-rose-800'
                      : analysisResult.classifier.predicted_label === 'feature'
                      ? 'bg-blue-950 text-blue-300 border border-blue-800'
                      : analysisResult.classifier.predicted_label === 'question'
                      ? 'bg-amber-950 text-amber-300 border border-amber-800'
                      : 'bg-gray-800 text-gray-400'
                  }`}>
                    {analysisResult.classifier.predicted_label}
                  </span>
                </div>
                <div className="flex items-center justify-between text-xs text-gray-400 mb-2">
                  <span>Confidence: {(analysisResult.classifier.confidence * 100).toFixed(1)}%</span>
                  {analysisResult.classifier.needs_review && (
                    <span className="text-amber-400 flex items-center gap-1">
                      <AlertTriangle className="w-3 h-3" /> Needs Review (uncertain)
                    </span>
                  )}
                </div>
                <p className="text-xs text-gray-400 italic">
                  Note: Pre-trained on NLBSE 2024 dataset. A threshold of 0.50 is enforced to filter ambiguous predictions.
                </p>
              </div>

              {/* LLM Synthesis */}
              <div className="p-4 rounded-lg bg-dark-900 border border-dark-700">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs text-gray-400 font-medium">Synthesized Type (Ollama Qwen 4B)</span>
                  <span className="px-2 py-0.5 rounded text-xs font-semibold uppercase bg-indigo-950 text-indigo-300 border border-indigo-800">
                    {analysisResult.synthesis.issue_type}
                  </span>
                </div>
                <p className="text-xs text-gray-300 leading-relaxed">
                  {analysisResult.synthesis.issue_type_reason}
                </p>
              </div>
            </div>
          </div>

          {/* Prominent Verified Claims Panel */}
          <div className="bg-dark-800 border border-dark-600 rounded-xl p-6 shadow-md">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-2 mb-4 pb-3 border-b border-dark-700">
              <div>
                <h3 className="text-base font-bold text-white flex items-center gap-2">
                  <ShieldCheck className="w-5 h-5 text-emerald-400" />
                  NLI Verified Claims Panel
                </h3>
                <p className="text-xs text-gray-400 mt-0.5">
                  Every claim is verified against retrieved documentation & issue chunks using DeBERTa-v3 NLI.
                </p>
              </div>

              <div className="flex items-center gap-3 text-xs">
                <span className="text-gray-400">
                  Thresholds: <code className="bg-dark-900 px-1.5 py-0.5 rounded text-gray-300">τ_s={analysisResult.verification.summary.thresholds_used.tau_support}</code>, <code className="bg-dark-900 px-1.5 py-0.5 rounded text-gray-300">τ_c={analysisResult.verification.summary.thresholds_used.tau_contradict}</code>
                </span>
                <span className="px-2 py-1 rounded bg-emerald-950/70 border border-emerald-800 text-emerald-300 font-medium">
                  {analysisResult.verification.summary.supported} / {analysisResult.verification.summary.total_claims} Supported
                </span>
              </div>
            </div>

            <div className="p-3 mb-4 rounded-lg bg-indigo-950/40 border border-indigo-900/60 text-xs text-indigo-300 flex items-start gap-2">
              <HelpCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
              <span>
                <strong>Verification Signal Disclaimer:</strong> Verdicts reflect automated natural language inference agreement against cited text excerpts, not formal mathematical proof. Always review expandable source premises for critical debugging decisions.
              </span>
            </div>

            {/* Claims List */}
            <div className="space-y-3">
              {analysisResult.verification.claims.map((claim) => {
                const isExpanded = expandedClaims[claim.id];
                return (
                  <div
                    key={claim.id}
                    className="p-4 rounded-lg bg-dark-900 border border-dark-700 transition-colors"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1.5">
                          {claim.verdict === 'SUPPORTED' ? (
                            <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-950 text-emerald-300 border border-emerald-800">
                              <CheckCircle className="w-3 h-3" /> SUPPORTED
                            </span>
                          ) : claim.verdict === 'CONTRADICTED' ? (
                            <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-rose-950 text-rose-300 border border-rose-800">
                              <XCircle className="w-3 h-3" /> CONTRADICTED
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-amber-950 text-amber-300 border border-amber-800">
                              <AlertTriangle className="w-3 h-3" /> INSUFFICIENT EVIDENCE
                            </span>
                          )}

                          <span className="text-xs text-gray-400 font-mono">
                            Cites: {claim.cited_evidence_ids.join(', ') || 'None'}
                          </span>
                        </div>

                        <p className="text-sm text-gray-200 font-medium leading-relaxed">
                          "{claim.claim_text}"
                        </p>

                        <div className="flex items-center gap-4 mt-2 text-xs text-gray-400 font-mono">
                          <span>p(entailment): <strong className="text-gray-200">{claim.entailment_prob.toFixed(4)}</strong></span>
                          <span>p(contradiction): <strong className="text-gray-200">{claim.contradiction_prob.toFixed(4)}</strong></span>
                          <span>p(neutral): <strong className="text-gray-200">{claim.neutral_prob.toFixed(4)}</strong></span>
                        </div>
                      </div>

                      <button
                        onClick={() => toggleClaimExpansion(claim.id)}
                        className="p-1 rounded text-gray-400 hover:text-white hover:bg-dark-700 transition-colors"
                        title="Toggle cited premise details"
                      >
                        {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                      </button>
                    </div>

                    {/* Expandable Winning Premise Detail */}
                    {isExpanded && claim.winning_premise && (
                      <div className="mt-3 pt-3 border-t border-dark-800 text-xs">
                        <div className="text-gray-400 font-medium mb-1">
                          Winning Premise ({claim.winning_premise.premise_type}):
                        </div>
                        <pre className="p-3 rounded bg-dark-950 text-gray-300 whitespace-pre-wrap font-mono text-xs border border-dark-800 overflow-x-auto max-h-48">
                          {claim.winning_premise.text}
                        </pre>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Structured Synthesis Panel */}
          <div className="bg-dark-800 border border-dark-600 rounded-xl p-6 shadow-md space-y-6">
            <div>
              <h4 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-2">Issue Title & Summary</h4>
              <h3 className="text-lg font-bold text-white mb-2">{analysisResult.synthesis.title}</h3>
              <p className="text-sm text-gray-300 leading-relaxed bg-dark-900 p-4 rounded-lg border border-dark-700">
                {analysisResult.synthesis.summary}
              </p>
            </div>

            <div>
              <h4 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-2">Technical Analysis & Root Cause</h4>
              <div className="text-sm text-gray-300 leading-relaxed bg-dark-900 p-4 rounded-lg border border-dark-700 whitespace-pre-wrap font-sans">
                {analysisResult.synthesis.technical_analysis}
              </div>
            </div>

            <div>
              <h4 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-2">Suggested Resolution</h4>
              <div className="text-sm text-gray-300 leading-relaxed bg-dark-900 p-4 rounded-lg border border-dark-700 whitespace-pre-wrap font-sans">
                {analysisResult.synthesis.suggested_resolution}
              </div>
            </div>
          </div>

          {/* Retrieved Evidence Cards */}
          <div className="bg-dark-800 border border-dark-600 rounded-xl p-6 shadow-md">
            <h3 className="text-base font-bold text-white mb-4 flex items-center gap-2">
              <BookOpen className="w-5 h-5 text-indigo-400" />
              Retrieved Evidence Items ({analysisResult.retrieval.count})
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {analysisResult.retrieval.evidence.map((item) => {
                const isExp = expandedEvidence[item.id];
                return (
                  <div key={item.id} className="p-4 rounded-lg bg-dark-900 border border-dark-700 flex flex-col justify-between">
                    <div>
                      <div className="flex items-center justify-between gap-2 mb-2">
                        <span className="px-2 py-0.5 rounded font-mono text-xs font-bold bg-indigo-950 text-indigo-300 border border-indigo-800">
                          {item.id}
                        </span>
                        <div className="flex items-center gap-2">
                          <span className={`px-2 py-0.5 rounded text-xs capitalize ${
                            item.source_type === 'issue'
                              ? 'bg-blue-950 text-blue-300'
                              : item.source_type === 'doc'
                              ? 'bg-purple-950 text-purple-300'
                              : 'bg-emerald-950 text-emerald-300'
                          }`}>
                            {item.source_type}
                          </span>
                          <span className="text-xs text-gray-400 font-mono">
                            score: {item.score.toFixed(3)}
                          </span>
                        </div>
                      </div>

                      <h4 className="text-sm font-semibold text-gray-200 mb-2 truncate" title={item.title || item.heading}>
                        {item.title || item.heading || 'Repository Content'}
                      </h4>

                      <p className={`text-xs text-gray-400 font-mono bg-dark-950 p-2.5 rounded border border-dark-800 leading-relaxed ${isExp ? '' : 'line-clamp-4'}`}>
                        {item.excerpt}
                      </p>
                    </div>

                    <div className="flex items-center justify-between pt-3 mt-3 border-t border-dark-800 text-xs">
                      <button
                        onClick={() => toggleEvidenceExpansion(item.id)}
                        className="text-gray-400 hover:text-white"
                      >
                        {isExp ? 'Show Less' : 'Show Full Excerpt'}
                      </button>

                      {item.url && (
                        <a
                          href={item.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-indigo-400 hover:text-indigo-300 flex items-center gap-1 font-medium"
                        >
                          View Source <ExternalLink className="w-3 h-3" />
                        </a>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Latency & Trace Footer */}
          <div className="p-4 rounded-xl bg-dark-800 border border-dark-600 flex flex-wrap items-center justify-between text-xs text-gray-400 gap-4">
            <div className="flex items-center gap-4">
              <span>Classifier: <strong className="text-gray-200">{analysisResult.timings.classifier_ms.toFixed(0)}ms</strong></span>
              <span>Retrieval: <strong className="text-gray-200">{analysisResult.timings.retrieval_ms.toFixed(0)}ms</strong></span>
              <span>Generation: <strong className="text-gray-200">{(analysisResult.timings.generation_ms / 1000).toFixed(1)}s</strong></span>
              <span>Verification: <strong className="text-gray-200">{(analysisResult.timings.verification_ms / 1000).toFixed(1)}s</strong></span>
              <span>Total Pipeline: <strong className="text-emerald-400">{(analysisResult.timings.total_pipeline_ms / 1000).toFixed(1)}s</strong></span>
            </div>
            <div className="font-mono text-gray-500">
              Trace: {analysisResult.trace_id}
            </div>
          </div>
        </div>
      )}

      {/* Search-Only Results View */}
      {searchOnlyResults && (
        <div className="bg-dark-800 border border-dark-600 rounded-xl p-6 shadow-md">
          <h3 className="text-base font-bold text-white mb-4 flex items-center gap-2">
            <Search className="w-5 h-5 text-indigo-400" />
            Hybrid Search Ranked Candidates ({searchOnlyResults.length})
          </h3>

          <div className="space-y-4">
            {searchOnlyResults.map((item, idx) => (
              <div key={item.chunk_id || idx} className="p-4 rounded-lg bg-dark-900 border border-dark-700">
                <div className="flex items-center justify-between gap-2 mb-2">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono px-2 py-0.5 rounded bg-dark-800 text-gray-300">
                      Rank #{idx + 1}
                    </span>
                    <span className="text-xs capitalize px-2 py-0.5 rounded bg-indigo-950 text-indigo-300">
                      {item.source_type}
                    </span>
                    <span className="text-xs text-gray-400 font-mono">
                      {item.repo}
                    </span>
                  </div>
                  <span className="text-xs font-mono text-gray-400">
                    score: {item.score.toFixed(4)}
                  </span>
                </div>

                <h4 className="text-sm font-semibold text-gray-200 mb-1">
                  {item.title || item.heading || item.file_path || `Issue #${item.issue_number}`}
                </h4>

                <p className="text-xs text-gray-400 font-mono bg-dark-950 p-2.5 rounded border border-dark-800 line-clamp-3">
                  {item.excerpt}
                </p>

                {item.url && (
                  <div className="pt-2 mt-2 border-t border-dark-800 flex justify-end">
                    <a
                      href={item.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-xs text-indigo-400 hover:text-indigo-300 flex items-center gap-1"
                    >
                      Open in GitHub <ExternalLink className="w-3 h-3" />
                    </a>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
