import { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import type { ChatMessage } from '../types/messages';
import { CodeBlock } from './CodeBlock';
import { ThinkingSection } from './ThinkingSection';
import { ImageViewer } from './ImageViewer';
import { ExecutionDetailDropdown } from './ExecutionDetailDropdown';

// Component for generating message with live timer
function GeneratingMessage({ message, timestamp }: { message: ChatMessage; timestamp: string }) {
  const [elapsed, setElapsed] = useState(0);
  const [reasoningExpanded, setReasoningExpanded] = useState(true);

  useEffect(() => {
    if (!message.isStreaming || !message.startTime) return;
    const interval = setInterval(() => {
      setElapsed(Math.floor((Date.now() - message.startTime!) / 1000));
    }, 1000);
    return () => clearInterval(interval);
  }, [message.isStreaming, message.startTime]);

  useEffect(() => {
    if (!message.isStreaming) setReasoningExpanded(false);
  }, [message.isStreaming]);

  const displayTime = message.endTime && message.startTime
    ? Math.floor((message.endTime - message.startTime) / 1000)
    : elapsed;

  const streamingContent = message.content || '';
  const streamingReasoning = message.reasoning || '';

  return (
    <div className="flex items-start gap-3 msg-enter">
      {/* AI Avatar */}
      <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-gradient-to-br from-accent to-accent-dark flex items-center justify-center">
        <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
        </svg>
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 text-xs text-text-tertiary mb-2 flex-wrap">
          <span>{timestamp}</span>
          {message.turnNumber !== undefined && message.turnNumber > 0 && (
            <span className="px-1.5 py-0.5 rounded bg-accent/10 text-accent text-xs font-display font-medium">
              Turn {message.turnNumber}
            </span>
          )}
          {message.modelUsed && (
            <span className="text-text-tertiary">{message.modelUsed}</span>
          )}
          <span className="text-accent font-display flex items-center gap-1">
            {message.isStreaming ? (
              <>
                <span className="inline-block w-1.5 h-1.5 rounded-full bg-accent animate-pulse" />
                Generating... {displayTime}s
              </>
            ) : (
              <>Generated in {displayTime}s</>
            )}
          </span>
        </div>

        {/* Reasoning section */}
        {streamingReasoning && (
          <div className="mb-3">
            <button
              onClick={() => setReasoningExpanded(!reasoningExpanded)}
              className="flex items-center gap-2 text-sm text-text-tertiary hover:text-text-primary transition-colors"
            >
              <svg className={`w-3 h-3 transition-transform ${reasoningExpanded ? 'rotate-90' : ''}`} fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clipRule="evenodd" />
              </svg>
              <svg className="w-4 h-4 text-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
              <span className="font-display font-medium">{message.isStreaming ? 'Thinking...' : 'Reasoning'}</span>
              {!message.isStreaming && (
                <span className="text-text-tertiary">({streamingReasoning.split(/\s+/).length} words)</span>
              )}
            </button>

            {reasoningExpanded && (
              <div className="mt-2 ml-5 pl-3 border-l-2 border-accent/20 bg-surface-raised/50 rounded-r-lg p-3">
                <div className="text-sm text-text-secondary leading-relaxed prose prose-sm max-w-none prose-invert">
                  <ReactMarkdown>{streamingReasoning}</ReactMarkdown>
                  {message.isStreaming && <span className="inline-block w-2 h-4 bg-accent cursor-blink ml-0.5 rounded-sm" />}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Streaming content */}
        {streamingContent ? (
          <div className="text-sm text-text-primary prose prose-sm max-w-none prose-invert">
            <ReactMarkdown>{streamingContent}</ReactMarkdown>
            {message.isStreaming && <span className="inline-block w-2 h-4 bg-accent cursor-blink ml-0.5 rounded-sm" />}
          </div>
        ) : message.isStreaming ? (
          <div className="text-sm text-text-tertiary">
            <span className="inline-block w-1.5 h-1.5 rounded-full bg-accent animate-pulse" />
          </div>
        ) : null}
      </div>
    </div>
  );
}

interface ChatMessageComponentProps {
  message: ChatMessage;
}

// Environment init message component
function EnvironmentInitMessage({ message, timestamp }: { message: ChatMessage; timestamp: string }) {
  const [descExpanded, setDescExpanded] = useState(false);
  const hasDescription = message.descriptionContent && message.initStatus === 'description_complete';

  return (
    <div className="py-1.5 msg-enter">
      <div className="flex items-center gap-2">
        {message.isExecuting ? (
          <span className="inline-block w-2 h-2 rounded-full bg-accent animate-pulse" />
        ) : (
          <span className="inline-block w-2 h-2 rounded-full bg-nv-green" />
        )}
        <span className="text-xs text-text-tertiary">{timestamp}</span>
        <span className={`text-sm ${message.isExecuting ? 'text-accent' : 'text-nv-green'}`}>
          {message.content}
        </span>
        {hasDescription && (
          <button
            onClick={() => setDescExpanded(!descExpanded)}
            className="ml-1 text-xs text-text-tertiary hover:text-text-primary transition-colors underline"
          >
            {descExpanded ? 'Hide' : 'Details'}
          </button>
        )}
      </div>
      {hasDescription && descExpanded && (
        <div className="mt-2 p-3 bg-surface-raised border border-surface-border rounded-lg text-sm text-text-secondary prose prose-sm max-w-none prose-invert">
          <ReactMarkdown>{message.descriptionContent || ''}</ReactMarkdown>
        </div>
      )}
    </div>
  );
}

// Helper to extract text before/after code blocks
function parseContentWithCode(content: string): { text: string; codeBlocks: string[] } {
  const codeBlockRegex = /```(?:python)?\s*([\s\S]*?)```/g;
  const codeBlocks: string[] = [];
  let match;
  while ((match = codeBlockRegex.exec(content)) !== null) {
    codeBlocks.push(match[1].trim());
  }
  const text = content.replace(codeBlockRegex, '').trim();
  return { text, codeBlocks };
}

export function ChatMessageComponent({ message }: ChatMessageComponentProps) {
  const timestamp = new Date(message.timestamp).toLocaleTimeString();

  switch (message.type) {
    case 'system':
      return (
        <div className="flex justify-center py-0.5 msg-enter">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-surface-raised border border-surface-border border-l-2 border-l-accent">
            <svg className="w-3 h-3 text-text-tertiary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span className="text-xs text-text-secondary">{message.content}</span>
            <span className="text-xs text-text-muted">{timestamp}</span>
          </div>
        </div>
      );

    case 'environment_init':
      return <EnvironmentInitMessage message={message} timestamp={timestamp} />;

    case 'stage_plan': {
      const data = message.progressData || {};
      const stages = Array.isArray(data.stages) ? data.stages as Record<string, unknown>[] : [];
      return (
        <div className="flex items-start gap-3 p-4 bg-blue-950/20 border border-blue-800/30 border-l-2 border-l-blue-400 rounded-lg msg-enter">
          <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-blue-500/10 flex items-center justify-center text-blue-400 font-display font-bold text-sm">
            S
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap mb-1">
              <span className="text-xs font-display uppercase tracking-wider text-blue-400">
                {message.progressEvent === 'plan_replanned' ? 'Replanned stages' : 'Stage plan'}
              </span>
              <span className="text-xs text-text-muted">{timestamp}</span>
            </div>
            <div className="text-sm text-text-primary font-medium mb-2">{message.content}</div>
            <div className="space-y-2">
              {stages.map((stage, index) => {
                const primitive = stage.primitive as Record<string, unknown> | undefined;
                return (
                  <div key={`${String(stage.stage_id)}-${index}`} className="p-2.5 bg-surface-sunken/70 border border-surface-border rounded">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-xs font-mono text-blue-400">{String(stage.stage_id ?? index + 1)}</span>
                      <span className="text-sm text-text-primary">{String(stage.objective ?? '')}</span>
                    </div>
                    <div className="mt-1 text-xs text-text-secondary font-mono">
                      {String(primitive?.action ?? '')} {String(primitive?.target ?? '')}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      );
    }

    case 'primitive_execution': {
      const steps = message.executionSteps || [];
      const target = message.primitiveTarget == null
        ? ''
        : typeof message.primitiveTarget === 'string'
        ? message.primitiveTarget
        : JSON.stringify(message.primitiveTarget);
      return (
        <div className={`flex items-start gap-3 p-3 border-l-2 rounded-r-lg msg-enter ${
          message.isExecuting ? 'border-l-accent bg-surface-raised/60' :
          message.success ? 'border-l-nv-green bg-nv-green/5' : 'border-l-red-500 bg-red-950/20'
        }`}>
          <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-surface-sunken flex items-center justify-center">
            {message.isExecuting ? (
              <span className="w-3 h-3 border-2 border-accent/30 border-t-accent rounded-full animate-spin" />
            ) : (
              <span className={`w-2 h-2 rounded-full ${message.success ? 'bg-nv-green' : 'bg-red-400'}`} />
            )}
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap mb-1.5">
              <span className="font-mono text-sm font-bold text-accent">{message.primitiveAction}</span>
              {target && <span className="text-sm text-text-primary break-all">{target}</span>}
              <span className="px-1.5 py-0.5 rounded bg-surface-overlay text-xs text-text-secondary">{message.turnRole}</span>
              <span className="text-xs text-text-muted">turn {(message.turnNumber ?? 0) + 1}</span>
              <span className="text-xs text-text-muted">stage {message.stageId}</span>
            </div>
            {message.primitiveParams && Object.keys(message.primitiveParams).length > 0 && (
              <pre className="mb-2 p-2 text-xs font-mono text-text-secondary bg-surface-sunken border border-surface-border rounded overflow-x-auto">
                {JSON.stringify(message.primitiveParams, null, 2)}
              </pre>
            )}
            {message.isExecuting && steps.length === 0 && (
              <div className="text-sm text-text-tertiary italic">Executing primitive...</div>
            )}
            {steps.length > 0 && (
              <ExecutionDetailDropdown
                steps={steps}
                blockIndex={message.turnNumber ?? 0}
                isExecuting={message.isExecuting}
                itemLabel="Turn"
              />
            )}
          </div>
        </div>
      );
    }

    case 'memory': {
      const initialized = message.progressEvent === 'memory_initialized';
      const memory = message.memorySnapshot || {};
      const taskMemory = memory.task_memory ? 'loaded' : 'empty';
      const failures = Array.isArray(memory.failure_history) ? memory.failure_history.length : 0;
      return (
        <div className="p-4 bg-purple-950/15 border border-purple-800/25 border-l-2 border-l-purple-400 rounded-lg msg-enter">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-xs font-display uppercase tracking-wider text-purple-400">
              {initialized ? 'Memory initialized' : 'Memory finalized'}
            </span>
            <span className="text-xs text-text-muted">{timestamp}</span>
            <span className="text-xs text-text-tertiary">task memory: {taskMemory}</span>
            <span className="text-xs text-text-tertiary">failures: {failures}</span>
          </div>
          <details className="mt-2 group">
            <summary className="text-xs text-text-secondary cursor-pointer hover:text-text-primary">View memory snapshot</summary>
            <pre className="mt-2 max-h-72 overflow-auto p-3 text-xs font-mono whitespace-pre-wrap text-text-secondary bg-surface-sunken border border-surface-border rounded">
              {JSON.stringify(memory, null, 2)}
            </pre>
          </details>
        </div>
      );
    }

    case 'trial_progress': {
      const event = message.progressEvent || 'progress';
      const data = message.progressData || {};
      const value = (key: string) => data[key] == null ? '' : String(data[key]);
      const titles: Record<string, string> = {
        stage_started: `Stage ${value('stage_id')} · ${value('objective')}`,
        tool_call: `${value('turn_role') || 'primary'} · ${value('action')} ${value('target')}`,
        stage_reward: `Reward ${value('reward')} · ${value('status')}`,
        recovery: `Recovery · ${value('action')}`,
        stage_completed: `Stage ${value('stage_id')} completed`,
        stage_failed: `Stage ${value('stage_id')} failed`,
        stage_aborted: `Stage ${value('stage_id')} aborted`,
        trial_finished: `Trial finished · ${value('stop_reason')}`,
      };
      const title = titles[event] || event.replaceAll('_', ' ');
      const detail = event === 'stage_reward'
        ? value('reward_code') || value('tool_code')
        : event === 'recovery'
        ? value('message') || value('failure_code')
        : event.startsWith('stage_')
        ? value('reason')
        : '';
      const positive = event === 'stage_completed' ||
        (event === 'stage_reward' && value('status').toLowerCase() === 'success') ||
        (event === 'trial_finished' && value('task_completed') === 'true');
      const negative = event.includes('failed') || event.includes('aborted') ||
        event.includes('exhausted') ||
        (event === 'stage_reward' && ['failure', 'unsafe'].includes(value('status').toLowerCase()));

      return (
        <div className={`flex items-start gap-3 py-2 px-3 rounded-md border-l-2 msg-enter ${
          positive ? 'bg-nv-green/5 border-l-nv-green' :
          negative ? 'bg-red-950/20 border-l-red-500' :
          'bg-surface-raised/60 border-l-accent/50'
        }`}>
          <div className={`mt-1.5 w-2 h-2 rounded-full flex-shrink-0 ${
            positive ? 'bg-nv-green' : negative ? 'bg-red-400' : 'bg-accent'
          }`} />
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-sm text-text-primary font-display font-medium capitalize">{title}</span>
              <span className="text-xs text-text-muted">turn {message.turnNumber ?? 0}</span>
              <span className="text-xs text-text-muted">{timestamp}</span>
            </div>
            {detail && <div className="mt-1 text-xs text-text-secondary break-words">{detail}</div>}
          </div>
        </div>
      );
    }

    case 'model_thinking':
      return (
        <div className="flex items-center gap-3 py-1.5 msg-enter">
          <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-gradient-to-br from-accent to-accent-dark flex items-center justify-center">
            <svg className="w-4 h-4 text-white animate-pulse" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <span className="text-accent animate-pulse font-display font-medium">Thinking...</span>
            <span className="text-text-tertiary text-xs">
              ({message.thinkingPhase === 'initial' ? 'generating code' : 'evaluating'})
            </span>
          </div>
        </div>
      );

    case 'model_streaming':
      return <GeneratingMessage message={message} timestamp={timestamp} />;

    case 'model_response': {
      let textContent = '';
      let codeBlocks = message.codeBlocks || [];

      if (message.content) {
        const parsed = parseContentWithCode(message.content);
        textContent = parsed.text;
        if (codeBlocks.length === 0 && parsed.codeBlocks.length > 0) {
          codeBlocks = parsed.codeBlocks;
        }
      }

      // When we have code blocks, extract only non-code reasoning from the text.
      // The raw content often contains the code mixed with comments — strip anything
      // that looks like Python code (lines starting with common code patterns).
      if (codeBlocks.length > 0 && textContent) {
        const lines = textContent.split('\n');
        const reasoningLines = lines.filter(line => {
          const trimmed = line.trim();
          if (!trimmed) return false;
          // Skip lines that look like Python code
          if (/^(import |from |def |class |if |for |while |return |print\(|#|[a-z_]\w*\s*[=\[\.(+])/i.test(trimmed)) return false;
          // Skip lines that are just variable names or function calls
          if (/^[a-z_]\w*(\.|$)/i.test(trimmed) && trimmed.length < 60) return false;
          return true;
        });
        textContent = reasoningLines.join('\n').trim();
      }

      const showText = textContent.length > 0;

      return (
        <div className="flex items-start gap-3 msg-enter border-l-2 border-l-accent/30 pl-2 bg-surface-raised/50 rounded-r-lg py-3 pr-3">
          {/* AI Avatar */}
          <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-gradient-to-br from-accent to-accent-dark flex items-center justify-center">
            <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 text-xs text-text-tertiary uppercase tracking-wider mb-2 flex-wrap">
              <span className="normal-case tracking-normal text-text-muted">{timestamp}</span>
              {message.decision && (
                <span className={`px-1.5 py-0.5 rounded text-xs font-display font-medium normal-case tracking-normal ${
                  message.decision === 'finish'
                    ? 'bg-nv-green/10 text-nv-green border border-nv-green/20'
                    : message.decision === 'regenerate'
                    ? 'bg-accent/10 text-accent border border-accent/20'
                    : 'bg-blue-900/20 text-blue-400 border border-blue-800/50'
                }`}>
                  {message.decision === 'finish' ? 'Finish' : message.decision === 'regenerate' ? 'Regenerate' : message.decision}
                </span>
              )}
            </div>

            {message.reasoning && <ThinkingSection content={message.reasoning} />}

            {showText && (
              <div className="text-sm text-text-primary mb-3 prose prose-sm max-w-none prose-invert">
                <ReactMarkdown>{textContent}</ReactMarkdown>
              </div>
            )}

            {codeBlocks.length > 0 && (
              <div className="space-y-2">
                {codeBlocks.map((code, idx) => (
                  <CodeBlock key={idx} code={code} collapsible defaultCollapsed />
                ))}
              </div>
            )}
          </div>
        </div>
      );
    }

    case 'code_execution': {
      const hasOutput = message.stdout || message.stderr;
      const hasExecutionSteps = message.executionSteps && message.executionSteps.length > 0;
      const execBorderColor = message.isExecuting
        ? 'border-l-accent'
        : message.success
        ? 'border-l-nv-green/50'
        : 'border-l-red-500/50';

      return (
        <div className={`flex items-start gap-3 msg-enter border-l-2 ${execBorderColor} pl-2`}>
          {/* Status dot */}
          <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-surface-sunken flex items-center justify-center">
            {message.isExecuting ? (
              <svg className="w-4 h-4 text-accent animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
            ) : (
              <span className={`inline-block w-2 h-2 rounded-full ${message.success ? 'bg-nv-green' : 'bg-red-400'}`} />
            )}
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 text-xs text-text-tertiary uppercase tracking-wider mb-1.5">
              <span className="font-mono text-accent">$</span>
              <span className="font-display">Block {(message.blockIndex ?? 0) + 1}</span>
              {message.isExecuting ? (
                <span className="text-accent font-display font-medium normal-case tracking-normal">Running...</span>
              ) : message.success ? (
                <span className="text-nv-green font-display font-medium normal-case tracking-normal">Completed</span>
              ) : (
                <span className="text-red-400 font-display font-medium normal-case tracking-normal">Failed</span>
              )}
              <span className="text-text-muted normal-case tracking-normal">{timestamp}</span>
            </div>

            {/* Execution Details Dropdown */}
            {hasExecutionSteps && (
              <ExecutionDetailDropdown
                steps={message.executionSteps!}
                blockIndex={message.blockIndex ?? 0}
                isExecuting={message.isExecuting}
              />
            )}

            {message.isExecuting && !hasExecutionSteps ? (
              <div className="text-sm text-text-tertiary italic">Executing code...</div>
            ) : hasOutput ? (
              <div className="space-y-2 mt-2">
                {message.stdout && (
                  <div className="text-xs">
                    <div className="text-xs text-text-tertiary uppercase tracking-wider font-display font-medium mb-1">stdout</div>
                    <pre className="bg-surface-sunken p-3 rounded-lg text-text-secondary overflow-x-auto font-mono text-xs whitespace-pre-wrap max-h-48 overflow-y-auto border border-surface-border">
                      {message.stdout}
                    </pre>
                  </div>
                )}
                {message.stderr && (
                  <div className="text-xs">
                    <div className="text-red-400 font-display font-medium mb-1">stderr</div>
                    <pre className="bg-red-950/30 p-3 rounded-lg text-red-400 overflow-x-auto font-mono text-xs whitespace-pre-wrap max-h-48 overflow-y-auto border border-red-800/20">
                      {message.stderr}
                    </pre>
                  </div>
                )}
              </div>
            ) : !message.isExecuting && !hasExecutionSteps ? (
              <div className="text-xs text-text-tertiary">No output</div>
            ) : null}
          </div>
        </div>
      );
    }

    case 'visual_feedback':
      return (
        <div className="flex items-start gap-3 msg-enter">
          <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-gradient-to-br from-accent to-accent-dark flex items-center justify-center">
            <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
          </div>
          <div className="flex-1">
            <div className="text-xs text-text-muted mb-2">{timestamp}</div>
            {message.content && (
              <div className="text-sm text-text-secondary mb-2">{message.content}</div>
            )}
            {message.imageBase64 && (
              <div className="border border-surface-border rounded-md overflow-hidden">
                <ImageViewer src={message.imageBase64} alt="Visual feedback" />
              </div>
            )}
          </div>
        </div>
      );

    case 'image_analysis':
      return (
        <div className="flex items-start gap-3 msg-enter">
          <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-gradient-to-br from-accent to-accent-dark flex items-center justify-center">
            <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
            </svg>
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 text-xs text-text-tertiary uppercase tracking-wider mb-2 flex-wrap">
              <span className="normal-case tracking-normal text-text-muted">{timestamp}</span>
              <span className="px-1.5 py-0.5 rounded bg-accent/10 text-accent border border-accent/20 font-display font-medium normal-case tracking-normal">
                {message.analysisType === 'state_comparison' ? 'State Comparison' : 'Initial Description'}
              </span>
              {message.modelUsed && (
                <span className="text-text-tertiary normal-case tracking-normal">via {message.modelUsed}</span>
              )}
            </div>
            <div className="text-sm text-text-primary bg-surface-raised rounded-lg p-4 border border-surface-border prose prose-sm max-w-none prose-invert">
              <ReactMarkdown>{message.content || ''}</ReactMarkdown>
            </div>
          </div>
        </div>
      );

    case 'user_prompt':
      return (
        <div className="flex items-start gap-3 justify-end msg-enter-right">
          <div className="max-w-[90%] sm:max-w-[70%]">
            <div className="text-xs text-text-muted mb-1 text-right">{timestamp}</div>
            <div className="bg-accent/8 border border-accent/15 text-text-primary rounded-xl rounded-br-sm px-4 py-3 text-sm break-words">
              {message.content}
            </div>
          </div>
        </div>
      );

    case 'completion': {
      const isSuccess = Boolean(message.success);
      return (
        <div className={`p-5 rounded-lg border msg-enter animate-scale-in ${
          isSuccess
            ? 'bg-nv-green/5 border-nv-green/20'
            : 'bg-red-950/30 border-red-800/20'
        }`}>
          <div className="flex items-center gap-3">
            <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
              isSuccess ? 'bg-gradient-to-br from-nv-green to-nv-green-light' : 'bg-red-600'
            }`}>
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                {isSuccess ? (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                ) : (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                )}
              </svg>
            </div>
            <div>
              <div className={`font-display font-bold text-sm ${isSuccess ? 'text-nv-green' : 'text-red-400'}`}>
                {isSuccess ? 'Trial Completed Successfully' : 'Trial Failed'}
              </div>
              <div className="text-xs text-text-tertiary mt-0.5">
                {isSuccess ? 'The model finished the task' : 'Maximum turns exceeded'}
              </div>
            </div>
          </div>
          {message.summary && (
            <details className="mt-3 group">
              <summary className="text-xs text-text-tertiary cursor-pointer list-none flex items-center gap-1 hover:text-text-primary transition-colors">
                <svg className="w-3 h-3 group-open:rotate-90 transition-transform" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clipRule="evenodd" />
                </svg>
                View full summary
              </summary>
              <pre className="mt-2 text-xs bg-surface-sunken p-3 rounded-lg overflow-x-auto whitespace-pre-wrap text-text-secondary border border-surface-border">
                {message.summary}
              </pre>
            </details>
          )}
        </div>
      );
    }

    case 'error':
      return (
        <div className="bg-red-950/30 border border-red-800/20 rounded-xl p-5 msg-enter animate-scale-in">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-red-600 flex items-center justify-center">
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
            </div>
            <div>
              <div className="font-display font-bold text-sm text-red-400">Error</div>
              <div className="text-sm text-red-400">{message.error}</div>
            </div>
          </div>
        </div>
      );

    default:
      return null;
  }
}
