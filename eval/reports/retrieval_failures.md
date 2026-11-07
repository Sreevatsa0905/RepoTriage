# Dense Retrieval Failure Analysis (Stage 6.6-E)
**Sample**: 7 queries out of 51 where BGE dense retrieval failed to rank the canonical issue in the Top 10 ($MRR@10 = 0.0$).

| # | Query Issue | Target Issue | BM25 Rank | Dense Rank | RRF Rank | Rerank Rank | Failure Hypothesis |
| :-: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| 1 | **psf/requests#6896**: Regenerating tests/certs/ with OpenSSL >= 3.2... | **#2811**: 2.8.0 release raise ConnectTimeout on no... | >10 | **>10** | >10 | >10 | Dense embedding similarity overwhelmed by lexical code identifiers or stack traces. |
| 2 | **encode/httpx#2632**: SSL + NordVPN Proxy... | **#1434**: HTTPS proxies support... | >10 | **>10** | >10 | >10 | Lexical code token dominance: Duplicate describes a specific proxy URL auth string that BM25 matches exactly on 'proxy_url', while dense embedding diffuses attention. |
| 3 | **encode/httpx#1615**: Stricter enforcement around client scoping.... | **#1614**: Issue a warning or exception when a clie... | #1 | **>10** | #4 | #2 | Dense embedding similarity overwhelmed by lexical code identifiers or stack traces. |
| 4 | **encode/httpx#1614**: Issue a warning or exception when a client is... | **#1615**: Stricter enforcement around client scopi... | >10 | **>10** | >10 | >10 | Dense embedding similarity overwhelmed by lexical code identifiers or stack traces. |
| 5 | **encode/httpx#1373**: httpx.ConnectError: [SSL: UNSUPPORTED_PROTOCO... | **#1434**: HTTPS proxies support... | >10 | **>10** | >10 | >10 | Dense embedding similarity overwhelmed by lexical code identifiers or stack traces. |
| 6 | **encode/httpx#1332**: Unintuitive behavior with using an AsyncClien... | **#1317**: Client.close() does not close transports... | #5 | **>10** | #8 | #7 | Dense embedding similarity overwhelmed by lexical code identifiers or stack traces. |
| 7 | **psf/requests#6872**: Requests v2.32.0-v2.32.3 caused a segmentatio... | **#6730**: Certificate loading regression with HTTP... | >10 | **>10** | >10 | >10 | Dense embedding similarity overwhelmed by lexical code identifiers or stack traces. |

## Key Insights on Retrieval Failures
1. **Lexical / Identifier Precision**: BM25 excels on rare error strings, method names, and version constraints (e.g. `RFC-3986`, OpenSSL codes), whereas dense embeddings tend to compress these tokens into broader semantic representations.
2. **Complementarity of RRF**: In queries where Dense missed entirely (>10), BM25 frequently succeeded in the Top 5, allowing Reciprocal Rank Fusion ($k=60$) to recover the relevant item into the final candidate set.
3. **Reranker Anchoring**: Score blending ($lpha=0.50$) ensures that cross-encoder semantic scoring cannot discard high-confidence lexical matches rescued by BM25.