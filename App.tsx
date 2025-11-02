
import React, { useState, useCallback, useEffect } from 'react';
import { Step } from './types';
import { TRAINING_STEPS, TESTING_STEPS } from './constants';
import { getExplanationForStep } from './services/geminiService';
import { StepSelector } from './components/StepSelector';
import { StepDetails } from './components/StepDetails';
import { GithubIcon } from './components/Icons';

export default function App(): React.ReactElement {
  const [activeStep, setActiveStep] = useState<Step>(TRAINING_STEPS[0]);
  const [explanation, setExplanation] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchExplanation = useCallback(async (step: Step) => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await getExplanationForStep(step);
      setExplanation(result);
    } catch (err) {
      console.error("Error fetching explanation:", err);
      setError("Failed to generate explanation. Please check your API key and try again.");
      setExplanation('');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchExplanation(activeStep);
  }, [activeStep, fetchExplanation]);

  return (
    <div className="min-h-screen bg-slate-900 font-sans">
      <header className="sticky top-0 z-10 bg-slate-900/70 backdrop-blur-md border-b border-slate-700">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <h1 className="text-xl sm:text-2xl font-bold text-white tracking-tight">
              PCA/Eigenfaces Interactive Explainer
            </h1>
            <a
              href="https://github.com/robaita/introduction_to_machine_learning/blob/main/dataset.zip"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center space-x-2 text-slate-400 hover:text-white transition-colors"
            >
              <GithubIcon />
              <span className="hidden sm:inline">Source Document</span>
            </a>
          </div>
        </div>
      </header>
      
      <main className="container mx-auto p-4 sm:p-6 lg:p-8">
        <div className="lg:grid lg:grid-cols-12 lg:gap-8">
          <aside className="lg:col-span-3 xl:col-span-2 mb-8 lg:mb-0">
            <nav className="sticky top-20">
              <StepSelector 
                title="Training Steps"
                steps={TRAINING_STEPS}
                activeStep={activeStep}
                onStepSelect={setActiveStep}
              />
              <div className="mt-8">
                <StepSelector 
                  title="Testing Steps"
                  steps={TESTING_STEPS}
                  activeStep={activeStep}
                  onStepSelect={setActiveStep}
                />
              </div>
            </nav>
          </aside>

          <div className="lg:col-span-9 xl:col-span-10">
            <StepDetails 
              step={activeStep}
              explanation={explanation}
              isLoading={isLoading}
              error={error}
            />
          </div>
        </div>
      </main>
    </div>
  );
}
