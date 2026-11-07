import React from 'react';
import { GitBranch, Cpu, Database, Activity } from 'lucide-react';
import type { SystemHealth } from '../types';

interface NavbarProps {
  activeTab: 'copilot' | 'evaluation';
  setActiveTab: (tab: 'copilot' | 'evaluation') => void;
  health: SystemHealth | null;
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, setActiveTab, health }) => {
  return (
    <header className="bg-dark-800 border-b border-dark-600 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <div className="flex items-center space-x-6">
          <div className="flex items-center space-x-3 cursor-pointer" onClick={() => setActiveTab('copilot')}>
            <div className="w-9 h-9 rounded-lg bg-indigo-600 flex items-center justify-center font-bold text-white shadow-md shadow-indigo-500/20">
              <GitBranch className="w-5 h-5" />
            </div>
            <div>
              <span className="text-lg font-bold tracking-tight text-white">RepoTriage</span>
              <span className="ml-2 text-xs px-2 py-0.5 rounded bg-indigo-900/60 text-indigo-300 border border-indigo-700/50">
                v1.0 Local Copilot
              </span>
            </div>
          </div>

          <nav className="flex space-x-1">
            <button
              onClick={() => setActiveTab('copilot')}
              className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${
                activeTab === 'copilot'
                  ? 'bg-dark-700 text-white shadow-sm'
                  : 'text-gray-400 hover:text-white hover:bg-dark-700/50'
              }`}
            >
              Issue Copilot
            </button>
            <button
              onClick={() => setActiveTab('evaluation')}
              className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${
                activeTab === 'evaluation'
                  ? 'bg-dark-700 text-white shadow-sm'
                  : 'text-gray-400 hover:text-white hover:bg-dark-700/50'
              }`}
            >
              Evaluation & Benchmarks
            </button>
          </nav>
        </div>

        <div className="flex items-center space-x-4 text-xs">
          {health?.gpu && (
            <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded-full bg-dark-700 border border-dark-600 text-gray-300">
              <Cpu className={`w-3.5 h-3.5 ${health.gpu.available ? 'text-emerald-400' : 'text-amber-400'}`} />
              <span>{health.gpu.available ? (health.gpu.device || 'NVIDIA GPU') : 'CPU Mode'}</span>
            </div>
          )}

          {health?.database && (
            <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded-full bg-dark-700 border border-dark-600 text-gray-300">
              <Database className="w-3.5 h-3.5 text-blue-400" />
              <span>{health.database.total_chunks ? health.database.total_chunks.toLocaleString() : 0} Chunks</span>
            </div>
          )}

          <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded-full bg-dark-700 border border-dark-600 text-gray-300">
            <Activity className="w-3.5 h-3.5 text-emerald-400" />
            <span>Ollama + DeBERTa</span>
          </div>
        </div>
      </div>
    </header>
  );
};
