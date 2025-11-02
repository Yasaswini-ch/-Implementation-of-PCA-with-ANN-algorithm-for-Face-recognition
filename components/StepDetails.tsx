import React from 'react';
import { Step } from '../types';
import { ImageGrid } from './ImageGrid';
import { InteractiveDemo } from './InteractiveDemo';
import { LoadingSpinner, ErrorIcon } from './Icons';
import { AccuracyGraph } from './AccuracyGraph';

interface StepDetailsProps {
  step: Step;
  explanation: string;
  isLoading: boolean;
  error: string | null;
}

// A simple utility to render markdown
const renderMarkdown = (markdown: string) => {
  if (typeof (window as any).marked === 'function') {
    return { __html: (window as any).marked.parse(markdown) };
  }
  return { __html: markdown.replace(/\n/g, '<br />') };
};

const Visualization: React.FC<{ step: Step }> = ({ step }) => {
    if (!step.hasVisual) return null;

    switch (step.visualType) {
        case 'dataset':
            return <ImageGrid key={step.id} title="Example Face Dataset" count={12} type="normal" />;
        case 'mean':
            return <ImageGrid key={step.id} title="Calculated 'Average' Face" count={1} type="mean" />;
        case 'mean-zero':
            return <ImageGrid key={step.id} title="Mean-Subtracted (Zero-Centered) Faces" count={12} type="mean-zero" />;
        case 'eigenfaces':
            return <ImageGrid key={step.id} title="Top Eigenfaces (Principal Components)" count={8} type="eigenface" />;
        case 'interactive':
            return <InteractiveDemo key={step.id} stepId={step.id} />;
        default:
            return null;
    }
}

export const StepDetails: React.FC<StepDetailsProps> = ({ step, explanation, isLoading, error }) => {
  return (
    <div className="bg-slate-800/50 rounded-lg shadow-lg p-6 lg:p-8 min-h-[60vh]">
      <h2 className="text-2xl font-bold text-white mb-2">{step.title}</h2>
      <p className="text-slate-400 italic mb-6 border-l-4 border-slate-600 pl-4">
        {step.description}
      </p>

      <div className="prose prose-invert prose-slate max-w-none prose-headings:text-sky-400 prose-a:text-sky-400 hover:prose-a:text-sky-300">
        <h3 className="text-lg font-semibold text-sky-400 border-b border-slate-700 pb-2 mb-4">
          AI-Powered Explanation
        </h3>
        {isLoading && (
          <div className="flex items-center justify-center py-10">
            <LoadingSpinner />
            <span className="ml-3 text-lg">Generating explanation...</span>
          </div>
        )}
        {error && (
            <div className="bg-red-900/50 border border-red-700 text-red-300 px-4 py-3 rounded-md flex items-center">
                <ErrorIcon />
                <span className="ml-3">{error}</span>
            </div>
        )}
        {!isLoading && !error && (
          <div dangerouslySetInnerHTML={renderMarkdown(explanation)} />
        )}
      </div>
      
      {step.id === 'train-6' && (
        <div className="mt-8">
            <AccuracyGraph />
        </div>
      )}

      <div className="mt-8">
        <Visualization step={step} />
      </div>
    </div>
  );
};