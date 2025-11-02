
import React from 'react';
import { Step } from '../types';

interface StepSelectorProps {
  title: string;
  steps: Step[];
  activeStep: Step;
  onStepSelect: (step: Step) => void;
}

export const StepSelector: React.FC<StepSelectorProps> = ({ title, steps, activeStep, onStepSelect }) => {
  return (
    <div>
      <h2 className="text-sm font-semibold text-sky-400 uppercase tracking-wider mb-3">{title}</h2>
      <ul className="space-y-1">
        {steps.map((step) => (
          <li key={step.id}>
            <button
              onClick={() => onStepSelect(step)}
              className={`w-full text-left px-3 py-2 rounded-md text-sm transition-colors duration-200 ${
                activeStep.id === step.id
                  ? 'bg-sky-500/20 text-white font-semibold'
                  : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
              }`}
            >
              {step.title}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
};
