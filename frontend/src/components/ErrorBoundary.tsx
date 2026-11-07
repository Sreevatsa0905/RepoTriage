import { Component, type ErrorInfo, type ReactNode } from 'react';
import { AlertOctagon, ChevronDown, ChevronUp, RefreshCw } from 'lucide-react';

interface Props {
  children: ReactNode;
  fallbackData?: any;
  onReset?: () => void;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  showRawJson: boolean;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
    errorInfo: null,
    showRawJson: false,
  };

  public static getDerivedStateFromError(error: Error): Partial<State> {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught an unhandled render error:', error, errorInfo);
    this.setState({ errorInfo });
  }

  private handleReset = () => {
    this.setState({ hasError: false, error: null, errorInfo: null, showRawJson: false });
    if (this.props.onReset) {
      this.props.onReset();
    }
  };

  public render() {
    if (this.state.hasError) {
      return (
        <div className="bg-dark-800 border border-red-800/80 rounded-xl p-6 shadow-xl space-y-4">
          <div className="flex items-start gap-3">
            <div className="p-2 rounded-lg bg-red-950/80 border border-red-800 text-red-400 flex-shrink-0">
              <AlertOctagon className="w-6 h-6" />
            </div>
            <div className="flex-1">
              <h3 className="text-base font-bold text-white">This result could not be displayed</h3>
              <p className="text-xs text-red-300 mt-1 font-mono bg-red-950/40 p-2.5 rounded border border-red-900/60 overflow-x-auto">
                {this.state.error?.message || 'Unknown render exception'}
              </p>
              <p className="text-xs text-gray-400 mt-2">
                A formatting or schema discrepancy prevented full rendering. The application state is preserved.
              </p>
            </div>
            {this.props.onReset && (
              <button
                onClick={this.handleReset}
                className="px-3 py-1.5 rounded-lg bg-dark-700 hover:bg-dark-600 text-xs text-gray-200 border border-dark-600 flex items-center gap-1.5 transition-colors"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                Reset View
              </button>
            )}
          </div>

          {this.props.fallbackData && (
            <div className="pt-2 border-t border-dark-700">
              <button
                onClick={() => this.setState((prev) => ({ showRawJson: !prev.showRawJson }))}
                className="text-xs text-indigo-400 hover:text-indigo-300 flex items-center gap-1 font-medium"
              >
                {this.state.showRawJson ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                {this.state.showRawJson ? 'Hide Raw Response JSON' : 'Inspect Raw Response JSON'}
              </button>

              {this.state.showRawJson && (
                <pre className="mt-3 p-4 rounded-lg bg-dark-950 border border-dark-700 text-gray-300 font-mono text-xs overflow-x-auto max-h-96 leading-relaxed">
                  {typeof this.props.fallbackData === 'string'
                    ? this.props.fallbackData
                    : JSON.stringify(this.props.fallbackData, null, 2)}
                </pre>
              )}
            </div>
          )}
        </div>
      );
    }

    return this.props.children;
  }
}
