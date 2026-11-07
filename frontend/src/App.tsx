import { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { IssueCopilot } from './components/IssueCopilot';
import { EvaluationTab } from './components/EvaluationTab';
import { ErrorBoundary } from './components/ErrorBoundary';
import type { SystemHealth, Repository } from './types';

export function App() {
  const [activeTab, setActiveTab] = useState<'copilot' | 'evaluation'>('copilot');
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [repositories, setRepositories] = useState<Repository[]>([]);

  useEffect(() => {
    // Fetch initial health and repositories
    fetch('/api/health')
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => data && setHealth(data))
      .catch((err) => console.warn('Health check failed', err));

    fetch('/api/repositories')
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => data?.repositories && setRepositories(data.repositories))
      .catch((err) => console.warn('Repositories fetch failed', err));
  }, []);

  return (
    <div className="min-h-screen bg-dark-900 text-gray-200 flex flex-col font-sans">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} health={health} />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <ErrorBoundary>
          {activeTab === 'copilot' ? (
            <IssueCopilot repositories={repositories} />
          ) : (
            <EvaluationTab />
          )}
        </ErrorBoundary>
      </main>

      <footer className="border-t border-dark-700 bg-dark-800/60 py-6 text-center text-xs text-gray-500">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <div>
            RepoTriage — Local GitHub Issue Intelligence & NLI Claim-Level Copilot
          </div>
          <div className="flex items-center space-x-4">
            <span>Local Offline-First</span>
            <span>•</span>
            <span>FastAPI + Vite React</span>
            <span>•</span>
            <span>RTX 3050 6GB</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
