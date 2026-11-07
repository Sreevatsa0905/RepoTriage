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
  BookOpen,
  Info
} from 'lucide-react';
import type { AnalyzeResponse, SearchResponse, Repository, EvidenceItem, VerifiedClaim } from '../types';
import { ErrorBoundary } from './ErrorBoundary';

interface IssueCopilotProps {
  repositories: Repository[];
}

export const IssueCopilot: React.FC<IssueCopilotProps> = ({ repositories = [] }) => {
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

  // Animated progress stepper during analysis
  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (isLoading) {
      setLoadingStep(1);
      const start = Date.now();
      interval = setInterval(() => {
        const elapsed = (Date.now() - start) / 1000;
        if (elapsed < 1.0) setLoadingStep(1);
        else if (elapsed < 3.0) setLoadingStep(2);
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
    setAnalysisResult(null);

    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: inputText,
          repo: selectedRepo,
          issue_url: issueUrl.trim() || null,
        }),
      });

      if (!res.ok) {
        if (res.status === 429) {
          throw new Error('Pipeline busy: another issue analysis is currently in progress. Please retry in a few moments.');
        }
        let detail = `Server error (${res.status})`;
        try {
          const errData = await res.json();
          if (errData?.detail) detail = errData.detail;
        } catch (_) {}
        throw new Error(detail);
      }

      const data: AnalyzeResponse = await res.json();
      setAnalysisResult(data);
    } catch (err: any) {
      console.error('Analyze request failed:', err);
      if (err.name === 'TypeError' && err.message?.includes('Failed to fetch')) {
        setErrorMsg('Unable to reach the RepoTriage API backend. Please ensure the backend server is running on http://127.0.0.1:8000.');
      } else {
        setErrorMsg(err.message || 'An unexpected error occurred during issue analysis.');
      }
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
    setSearchOnlyResults(null);

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
        let detail = `Search failed (${res.status})`;
        try {
          const errData = await res.json();
          if (errData?.detail) detail = errData.detail;
        } catch (_) {}
        throw new Error(detail);
      }

      const data: SearchResponse = await res.json();
      setSearchOnlyResults(Array.isArray(data?.results) ? data.results : []);
    } catch (err: any) {
      console.error('Search request failed:', err);
      if (err.name === 'TypeError' && err.message?.includes('Failed to fetch')) {
        setErrorMsg('Unable to reach the RepoTriage API backend. Please ensure the backend server is running on http://127.0.0.1:8000.');
      } else {
        setErrorMsg(err.message || 'Failed to execute hybrid search.');
      }
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

  // Safe data extraction helpers
  const classification = analysisResult?.classification || analysisResult?.classifier || {};
  const predictedLabel = (classification.label || classification.predicted_category || 'uncertain').toLowerCase();
  const confidenceScore = typeof classification.confidence === 'number' ? classification.confidence : 0;
  const needsReview = Boolean(classification.needs_review);

  const analysis = analysisResult?.analysis || analysisResult?.synthesis || {};
  const synthesizedType = (analysis.issue_type || 'unspecified').toLowerCase();
  const issueSummary = analysis.issue_summary || 'No summary synthesized';

  // Defensive claim normalization
  const claimsList: VerifiedClaim[] = Array.isArray(analysisResult?.verified_claims)
    ? analysisResult!.verified_claims
    : Array.isArray(analysisResult?.verification?.claims)
    ? analysisResult!.verification!.claims
    : Array.isArray(analysis.claims)
    ? (analysis.claims as any[])
    : [];

  const supportedClaimsCount = claimsList.filter((c) => c?.verdict === 'SUPPORTED').length;

  // Defensive evidence items normalization
  const evidenceList: EvidenceItem[] = Array.isArray(analysisResult?.evidence_items)
    ? analysisResult!.evidence_items
    : Array.isArray(analysisResult?.similar_issues)
    ? [...analysisResult!.similar_issues, ...(analysisResult?.doc_excerpts || [])]
    : [];

  // Defensive timing metrics
  const timings = analysisResult?.timings_ms || analysisResult?.timings || {};
  const traceIdentifier = analysisResult?.trace_file
    ? analysisResult.trace_file.split('/').pop()?.replace('.json', '')
    : analysisResult?.trace_id || 'trace-active';

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
            <div className="p-3.5 rounded-lg bg-red-950/70 border border-red-800 text-red-200 text-xs flex items-start gap-2.5">
              <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5 text-red-400" />
              <div className="flex-1 leading-relaxed">{errorMsg}</div>
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

      {/* Loading Progress Stepper */}
      {isLoading && (
        <div className="bg-dark-800 border border-indigo-900/50 rounded-xl p-6 shadow-xl animate-pulse">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-base font-semibold text-white flex items-center gap-2">
              <Clock className="w-5 h-5 text-indigo-400 animate-spin" />
              Executing Multi-Stage Triage Pipeline
            </h3>
            <span className="text-xs text-indigo-300 bg-indigo-950/80 px-2.5 py-1 rounded-full border border-indigo-800">
              Estimated duration: ~30-45s (local hardware)
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

      {/* Analysis Results View Wrapped in ErrorBoundary */}
      {analysisResult && (
        <ErrorBoundary fallbackData={analysisResult} onReset={() => setAnalysisResult(null)}>
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
                    <span
                      className={`px-2 py-0.5 rounded text-xs font-semibold uppercase ${
                        predictedLabel === 'bug'
                          ? 'bg-rose-950 text-rose-300 border border-rose-800'
                          : predictedLabel === 'feature'
                          ? 'bg-blue-950 text-blue-300 border border-blue-800'
                          : predictedLabel === 'question'
                          ? 'bg-amber-950 text-amber-300 border border-amber-800'
                          : 'bg-gray-800 text-gray-400 border border-gray-700'
                      }`}
                    >
                      {predictedLabel}
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-xs text-gray-400 mb-2">
                    <span>Confidence: {(confidenceScore * 100).toFixed(1)}%</span>
                    {needsReview && (
                      <span className="text-amber-400 flex items-center gap-1 font-medium">
                        <AlertTriangle className="w-3 h-3" /> Needs Review (uncertain)
                      </span>
                    )}
                  </div>
                  <p className="text-[11px] text-gray-400 italic">
                    Trained on NLBSE 2024 dataset. A threshold of 0.50 is enforced to filter ambiguous predictions.
                  </p>
                </div>

                {/* LLM Synthesis */}
                <div className="p-4 rounded-lg bg-dark-900 border border-dark-700">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs text-gray-400 font-medium">Synthesized Type (Ollama Qwen 4B)</span>
                    <span className="px-2 py-0.5 rounded text-xs font-semibold uppercase bg-indigo-950 text-indigo-300 border border-indigo-800">
                      {synthesizedType}
                    </span>
                  </div>
                  <p className="text-xs text-gray-300 leading-relaxed">
                    {issueSummary}
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
                    Thresholds: <code className="bg-dark-900 px-1.5 py-0.5 rounded text-gray-300">τ_s=0.80</code>, <code className="bg-dark-900 px-1.5 py-0.5 rounded text-gray-300">τ_c=0.80</code>
                  </span>
                  <span className="px-2 py-1 rounded bg-emerald-950/70 border border-emerald-800 text-emerald-300 font-medium">
                    {supportedClaimsCount} / {claimsList.length} Supported
                  </span>
                </div>
              </div>

              <div className="p-3 mb-4 rounded-lg bg-indigo-950/40 border border-indigo-900/60 text-xs text-indigo-300 flex items-start gap-2">
                <HelpCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                <span>
                  <strong>Verification Signal Disclaimer:</strong> Verdicts reflect automated natural language inference agreement against cited text excerpts, not mathematical proof. Always review expandable source premises for critical debugging decisions.
                </span>
              </div>

              {/* Claims List */}
              {claimsList.length === 0 ? (
                <div className="p-4 rounded-lg bg-dark-900 border border-dark-700 text-xs text-gray-400 text-center">
                  No discrete factual claims were extracted for verification.
                </div>
              ) : (
                <div className="space-y-3">
                  {claimsList.map((claim, idx) => {
                    const cid = claim.claim_id || claim.id || `claim_${idx + 1}`;
                    const isExpanded = Boolean(expandedClaims[cid]);
                    const verdict = claim.verdict || 'INSUFFICIENT_EVIDENCE';
                    const claimText = claim.text || claim.claim_text || 'Factual claim text';
                    const citedIds = claim.evidence_ids || claim.cited_evidence_ids || [];

                    const pEntailment = typeof claim.entailment_score === 'number'
                      ? claim.entailment_score
                      : typeof claim.entailment_prob === 'number'
                      ? claim.entailment_prob
                      : 0;

                    const pContradiction = typeof claim.contradiction_score === 'number'
                      ? claim.contradiction_score
                      : typeof claim.contradiction_prob === 'number'
                      ? claim.contradiction_prob
                      : 0;

                    const pNeutral = typeof claim.neutral_score === 'number'
                      ? claim.neutral_score
                      : typeof claim.neutral_prob === 'number'
                      ? claim.neutral_prob
                      : 0;

                    const winningExcerpt = claim.best_excerpt || claim.winning_premise?.text || '';
                    const winningType = claim.best_premise_type || claim.winning_premise?.premise_type || 'passage';

                    return (
                      <div
                        key={cid}
                        className="p-4 rounded-lg bg-dark-900 border border-dark-700 transition-colors"
                      >
                        <div className="flex items-start justify-between gap-3">
                          <div className="flex-1">
                            <div className="flex items-center gap-2 mb-1.5">
                              {verdict === 'SUPPORTED' ? (
                                <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-950 text-emerald-300 border border-emerald-800">
                                  <CheckCircle className="w-3 h-3" /> SUPPORTED
                                </span>
                              ) : verdict === 'CONTRADICTED' ? (
                                <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-rose-950 text-rose-300 border border-rose-800">
                                  <XCircle className="w-3 h-3" /> CONTRADICTED
                                </span>
                              ) : (
                                <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-amber-950 text-amber-300 border border-amber-800">
                                  <AlertTriangle className="w-3 h-3" /> INSUFFICIENT EVIDENCE
                                </span>
                              )}

                              <span className="text-xs text-gray-400 font-mono">
                                Cites: {citedIds.join(', ') || 'None'}
                              </span>
                            </div>

                            <p className="text-sm text-gray-200 font-medium leading-relaxed">
                              "{claimText}"
                            </p>

                            <div className="flex items-center gap-4 mt-2 text-xs text-gray-400 font-mono">
                              <span>p(entailment): <strong className="text-gray-200">{pEntailment.toFixed(4)}</strong></span>
                              <span>p(contradiction): <strong className="text-gray-200">{pContradiction.toFixed(4)}</strong></span>
                              <span>p(neutral): <strong className="text-gray-200">{pNeutral.toFixed(4)}</strong></span>
                            </div>
                          </div>

                          {winningExcerpt && (
                            <button
                              onClick={() => toggleClaimExpansion(cid)}
                              className="p-1 rounded text-gray-400 hover:text-white hover:bg-dark-700 transition-colors"
                              title="Toggle cited premise details"
                            >
                              {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                            </button>
                          )}
                        </div>

                        {/* Expandable Winning Premise Detail */}
                        {isExpanded && winningExcerpt && (
                          <div className="mt-3 pt-3 border-t border-dark-800 text-xs">
                            <div className="text-gray-400 font-medium mb-1">
                              Winning Premise ({winningType}):
                            </div>
                            <pre className="p-3 rounded bg-dark-950 text-gray-300 whitespace-pre-wrap font-mono text-xs border border-dark-800 overflow-x-auto max-h-48">
                              {winningExcerpt}
                            </pre>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            {/* Structured Synthesis Panel */}
            <div className="bg-dark-800 border border-dark-600 rounded-xl p-6 shadow-md space-y-5">
              <div>
                <h4 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-2">Issue Summary</h4>
                <p className="text-sm text-gray-300 leading-relaxed bg-dark-900 p-4 rounded-lg border border-dark-700">
                  {issueSummary}
                </p>
              </div>

              {Array.isArray(analysis.likely_causes) && analysis.likely_causes.length > 0 && (
                <div>
                  <h4 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-2">Likely Root Causes</h4>
                  <ul className="text-sm text-gray-300 leading-relaxed bg-dark-900 p-4 rounded-lg border border-dark-700 space-y-1 list-disc list-inside">
                    {analysis.likely_causes.map((c: string, i: number) => (
                      <li key={i}>{c}</li>
                    ))}
                  </ul>
                </div>
              )}

              {Array.isArray(analysis.recommended_next_steps) && analysis.recommended_next_steps.length > 0 && (
                <div>
                  <h4 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-2">Recommended Next Steps</h4>
                  <ul className="text-sm text-gray-300 leading-relaxed bg-dark-900 p-4 rounded-lg border border-dark-700 space-y-1 list-disc list-inside">
                    {analysis.recommended_next_steps.map((s: string, i: number) => (
                      <li key={i}>{s}</li>
                    ))}
                  </ul>
                </div>
              )}

              {analysis.limitations && (
                <div>
                  <h4 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-1 flex items-center gap-1.5">
                    <Info className="w-3.5 h-3.5 text-gray-400" /> Limitations & Missing Information
                  </h4>
                  <p className="text-xs text-gray-400 italic bg-dark-900 p-3 rounded-lg border border-dark-700">
                    {analysis.limitations}
                  </p>
                </div>
              )}
            </div>

            {/* Retrieved Evidence Cards */}
            <div className="bg-dark-800 border border-dark-600 rounded-xl p-6 shadow-md">
              <h3 className="text-base font-bold text-white mb-4 flex items-center gap-2">
                <BookOpen className="w-5 h-5 text-indigo-400" />
                Retrieved Evidence Items ({evidenceList.length})
              </h3>

              {evidenceList.length === 0 ? (
                <div className="p-4 rounded-lg bg-dark-900 border border-dark-700 text-xs text-gray-400 text-center">
                  No evidence chunks retrieved for this query.
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {evidenceList.map((item, idx) => {
                    const eid = item.evidence_id || item.id || item.chunk_id || `E${idx + 1}`;
                    const isExp = Boolean(expandedEvidence[eid]);
                    const itemType = item.source_type || 'issue';
                    const itemTitle = item.heading || item.title || item.file_path || (item.issue_number ? `Issue #${item.issue_number}` : 'Reference passage');
                    const itemScore = typeof item.blend_score === 'number'
                      ? item.blend_score
                      : typeof item.final_score === 'number'
                      ? item.final_score
                      : typeof item.score === 'number'
                      ? item.score
                      : 0;
                    const itemExcerpt = item.text || item.excerpt || 'No excerpt available';

                    return (
                      <div key={eid} className="p-4 rounded-lg bg-dark-900 border border-dark-700 flex flex-col justify-between">
                        <div>
                          <div className="flex items-center justify-between gap-2 mb-2">
                            <span className="px-2 py-0.5 rounded font-mono text-xs font-bold bg-indigo-950 text-indigo-300 border border-indigo-800">
                              {eid}
                            </span>
                            <div className="flex items-center gap-2">
                              <span
                                className={`px-2 py-0.5 rounded text-xs capitalize ${
                                  itemType === 'issue'
                                    ? 'bg-blue-950 text-blue-300'
                                    : itemType === 'doc'
                                    ? 'bg-purple-950 text-purple-300'
                                    : 'bg-emerald-950 text-emerald-300'
                                }`}
                              >
                                {itemType}
                              </span>
                              <span className="text-xs text-gray-400 font-mono">
                                score: {itemScore.toFixed(3)}
                              </span>
                            </div>
                          </div>

                          <h4 className="text-sm font-semibold text-gray-200 mb-2 truncate" title={itemTitle}>
                            {itemTitle}
                          </h4>

                          <p className={`text-xs text-gray-400 font-mono bg-dark-950 p-2.5 rounded border border-dark-800 leading-relaxed ${isExp ? '' : 'line-clamp-4'}`}>
                            {itemExcerpt}
                          </p>
                        </div>

                        <div className="flex items-center justify-between pt-3 mt-3 border-t border-dark-800 text-xs">
                          <button
                            onClick={() => toggleEvidenceExpansion(eid)}
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
              )}
            </div>

            {/* Latency & Trace Footer */}
            <div className="p-4 rounded-xl bg-dark-800 border border-dark-600 flex flex-wrap items-center justify-between text-xs text-gray-400 gap-4">
              <div className="flex items-center gap-4">
                <span>
                  Classifier: <strong className="text-gray-200">{typeof timings.classifier_ms === 'number' ? timings.classifier_ms.toFixed(0) : '0'}ms</strong>
                </span>
                <span>
                  Retrieval: <strong className="text-gray-200">{typeof timings.retrieval_ms === 'number' ? timings.retrieval_ms.toFixed(0) : '0'}ms</strong>
                </span>
                <span>
                  Generation: <strong className="text-gray-200">{typeof timings.generation_ms === 'number' ? (timings.generation_ms / 1000).toFixed(1) : '0.0'}s</strong>
                </span>
                <span>
                  Verification: <strong className="text-gray-200">{typeof (timings.guardrail_ms ?? timings.verification_ms) === 'number' ? ((timings.guardrail_ms ?? timings.verification_ms)! / 1000).toFixed(1) : '0.0'}s</strong>
                </span>
                <span>
                  Total Pipeline: <strong className="text-emerald-400">{typeof (timings.total_ms ?? timings.total_pipeline_ms) === 'number' ? ((timings.total_ms ?? timings.total_pipeline_ms)! / 1000).toFixed(1) : '0.0'}s</strong>
                </span>
              </div>
              <div className="font-mono text-gray-500">
                Trace: {traceIdentifier}
              </div>
            </div>
          </div>
        </ErrorBoundary>
      )}

      {/* Search-Only Results View */}
      {searchOnlyResults && (
        <div className="bg-dark-800 border border-dark-600 rounded-xl p-6 shadow-md">
          <h3 className="text-base font-bold text-white mb-4 flex items-center gap-2">
            <Search className="w-5 h-5 text-indigo-400" />
            Hybrid Search Ranked Candidates ({searchOnlyResults.length})
          </h3>

          {searchOnlyResults.length === 0 ? (
            <div className="p-4 rounded-lg bg-dark-900 border border-dark-700 text-xs text-gray-400 text-center">
              No matching chunks found.
            </div>
          ) : (
            <div className="space-y-4">
              {searchOnlyResults.map((item, idx) => (
                <div key={item.chunk_id || idx} className="p-4 rounded-lg bg-dark-900 border border-dark-700">
                  <div className="flex items-center justify-between gap-2 mb-2">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono px-2 py-0.5 rounded bg-dark-800 text-gray-300">
                        Rank #{idx + 1}
                      </span>
                      <span className="text-xs capitalize px-2 py-0.5 rounded bg-indigo-950 text-indigo-300">
                        {item.source_type || 'chunk'}
                      </span>
                      <span className="text-xs text-gray-400 font-mono">
                        {item.repo || selectedRepo}
                      </span>
                    </div>
                    <span className="text-xs font-mono text-gray-400">
                      score: {(typeof item.score === 'number' ? item.score : (item.final_score ?? 0)).toFixed(4)}
                    </span>
                  </div>

                  <h4 className="text-sm font-semibold text-gray-200 mb-1">
                    {item.heading || item.title || item.file_path || (item.issue_number ? `Issue #${item.issue_number}` : 'Search candidate')}
                  </h4>

                  <p className="text-xs text-gray-400 font-mono bg-dark-950 p-2.5 rounded border border-dark-800 line-clamp-3">
                    {item.text || item.excerpt || 'No text content available'}
                  </p>

                  {item.url && (
                    <div className="pt-2 mt-2 border-t border-dark-800 flex justify-end">
                      <a
                        href={item.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-xs text-indigo-400 hover:text-indigo-300 flex items-center gap-1 font-medium"
                      >
                        Open in GitHub <ExternalLink className="w-3 h-3" />
                      </a>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
