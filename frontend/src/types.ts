export interface Repository {
  repo: string;
  issue_chunks: number;
  comment_chunks: number;
  doc_chunks: number;
  total_chunks: number;
}

export interface EvidenceItem {
  id: string;
  chunk_id: string;
  source_type: 'issue' | 'comment' | 'doc';
  repo: string;
  issue_number?: number;
  file_path?: string;
  heading?: string;
  title?: string;
  url: string;
  score: number;
  excerpt: string;
}

export interface VerifiedClaim {
  id: string;
  claim_text: string;
  cited_evidence_ids: string[];
  verdict: 'SUPPORTED' | 'CONTRADICTED' | 'INSUFFICIENT_EVIDENCE';
  entailment_prob: number;
  contradiction_prob: number;
  neutral_prob: number;
  winning_premise?: {
    premise_type: string;
    text: string;
    cited_ids?: string[];
  };
  explanation?: string;
}

export interface AnalyzeResponse {
  query: string;
  repo: string;
  classifier: {
    predicted_label: string;
    confidence: number;
    probabilities: Record<string, number>;
    needs_review: boolean;
    reason: string;
  };
  retrieval: {
    count: number;
    evidence: EvidenceItem[];
  };
  synthesis: {
    title: string;
    issue_type: string;
    issue_type_reason: string;
    summary: string;
    claims: Array<{
      id: string;
      claim_text: string;
      cited_evidence_ids: string[];
    }>;
    technical_analysis: string;
    suggested_resolution: string;
  };
  verification: {
    claims: VerifiedClaim[];
    summary: {
      total_claims: number;
      supported: number;
      contradicted: number;
      insufficient_evidence: number;
      thresholds_used: {
        tau_support: number;
        tau_contradict: number;
      };
    };
  };
  timings: {
    classifier_ms: number;
    retrieval_ms: number;
    generation_ms: number;
    verification_ms: number;
    total_pipeline_ms: number;
  };
  trace_id: string;
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
  gpu: {
    available: boolean;
    device: string | null;
  };
  database: {
    exists: boolean;
    total_chunks: number;
    repositories: Record<string, number>;
  };
  models: Record<string, string>;
}
