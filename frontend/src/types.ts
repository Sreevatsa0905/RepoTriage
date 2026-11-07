export interface Repository {
  repo: string;
  issue_chunks: number;
  comment_chunks: number;
  doc_chunks: number;
  total_chunks: number;
}

export interface EvidenceItem {
  evidence_id?: string;
  id?: string;
  chunk_id?: string;
  repo?: string;
  source_type?: 'issue' | 'comment' | 'doc';
  issue_number?: number | null;
  file_path?: string | null;
  heading?: string | null;
  title?: string | null;
  url?: string;
  text?: string;
  excerpt?: string;
  score?: number;
  blend_score?: number;
  final_score?: number;
  rerank_score?: number;
}

export interface VerifiedClaim {
  claim_id?: string;
  id?: string;
  text?: string;
  claim_text?: string;
  evidence_ids?: string[];
  cited_evidence_ids?: string[];
  verdict?: 'SUPPORTED' | 'CONTRADICTED' | 'INSUFFICIENT_EVIDENCE' | string;
  entailment_score?: number;
  contradiction_score?: number;
  neutral_score?: number;
  entailment_prob?: number;
  contradiction_prob?: number;
  neutral_prob?: number;
  best_evidence_id?: string;
  best_excerpt?: string;
  best_premise_type?: string;
  winning_premise?: {
    premise_type?: string;
    text?: string;
  };
}

export interface ClassificationResult {
  label?: string;
  predicted_category?: string;
  confidence?: number;
  scores?: Record<string, number>;
  threshold?: number;
  needs_review?: boolean;
  model_version?: string;
  benchmark_metrics?: {
    pooled_test_macro_f1?: number;
    cold_start_macro_f1?: number;
  };
}

export interface AnalysisSynthesis {
  issue_summary?: string;
  issue_type?: string;
  affected_components?: string[];
  observed_symptoms?: string[];
  environment_details?: string;
  reproduction_steps?: string[];
  missing_information?: string[];
  likely_causes?: string[];
  recommended_diagnostics?: string[];
  recommended_next_steps?: string[];
  claims?: Array<{
    claim_id?: string;
    text?: string;
    evidence_ids?: string[];
  }>;
  limitations?: string;
}

export interface TimingsMs {
  classifier_ms?: number;
  retrieval_ms?: number;
  generation_ms?: number;
  guardrail_ms?: number;
  verification_ms?: number;
  total_ms?: number;
  total_pipeline_ms?: number;
}

export interface AnalyzeResponse {
  query?: {
    text?: string;
    repo?: string;
    issue_url?: string | null;
  };
  classification?: ClassificationResult;
  classifier?: ClassificationResult; // fallback compatibility
  similar_issues?: EvidenceItem[];
  doc_excerpts?: EvidenceItem[];
  evidence_items?: EvidenceItem[];
  analysis?: AnalysisSynthesis;
  synthesis?: AnalysisSynthesis; // fallback compatibility
  verified_claims?: VerifiedClaim[];
  verification?: {
    claims?: VerifiedClaim[];
    summary?: {
      total_claims?: number;
      supported?: number;
      contradicted?: number;
      insufficient_evidence?: number;
      thresholds_used?: {
        tau_support?: number;
        tau_contradict?: number;
      };
    };
  };
  timings_ms?: TimingsMs;
  timings?: TimingsMs; // fallback compatibility
  generation_trace?: Record<string, any>;
  timestamp?: string;
  trace_file?: string;
  trace_id?: string;
}

export interface SearchResponse {
  query: string;
  repo: string | null;
  method: string;
  count: number;
  results: EvidenceItem[];
}

export interface SystemHealth {
  status: string;
  service: string;
  timestamp: string;
  gpu?: {
    available?: boolean;
    device?: string | null;
  };
  database?: {
    exists?: boolean;
    total_chunks?: number;
    repositories?: Record<string, number>;
  };
  models?: Record<string, string>;
}
