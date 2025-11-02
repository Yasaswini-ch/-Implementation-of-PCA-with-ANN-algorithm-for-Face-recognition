import React, { useState, useRef } from 'react';
import { UploadIcon, LoadingSpinner, ArrowRightIcon, QuestionMarkCircleIcon } from './Icons';

interface InteractiveDemoProps {
    stepId: string;
}

const UploadPlaceholder: React.FC<{ onImageUpload: (file: File) => void }> = ({ onImageUpload }) => {
    const inputRef = useRef<HTMLInputElement>(null);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            onImageUpload(e.target.files[0]);
        }
    };
    
    const handleClick = () => inputRef.current?.click();

    return (
        <div 
            className="w-full max-w-xs mx-auto border-2 border-dashed border-slate-600 rounded-lg p-8 flex flex-col items-center justify-center text-slate-400 cursor-pointer hover:border-sky-500 hover:text-sky-400 transition-colors"
            onClick={handleClick}
        >
            <input type="file" accept="image/*" ref={inputRef} onChange={handleFileChange} className="hidden" />
            <UploadIcon />
            <p className="mt-2 text-sm font-semibold">Upload Test Image</p>
        </div>
    );
};

const ResultDisplay: React.FC<{ uploadedImage: string; isImposter: boolean }> = ({ uploadedImage, isImposter }) => {
    return (
        <div className="flex flex-col sm:flex-row items-center justify-center gap-4 sm:gap-8">
            <div className="text-center">
                <img src={uploadedImage} alt="Uploaded" className="w-32 h-32 rounded-lg object-cover mx-auto shadow-lg" />
                <p className="mt-2 text-sm font-semibold text-slate-300">Your Image</p>
            </div>
            <div className="text-sky-400 my-4 sm:my-0">
                <ArrowRightIcon />
            </div>
            {isImposter ? (
                <div className="text-center">
                    <div className="w-32 h-32 rounded-lg bg-slate-700 flex flex-col items-center justify-center shadow-lg text-amber-400">
                        <QuestionMarkCircleIcon />
                    </div>
                    <p className="mt-2 text-sm font-semibold text-amber-400">Unknown Person</p>
                    <p className="text-xs text-slate-400">(Imposter Detected)</p>
                </div>
            ) : (
                <div className="text-center">
                    <div className="w-32 h-32 rounded-lg bg-slate-700 flex items-center justify-center shadow-lg">
                        <img src="https://picsum.photos/seed/dataset5/200" alt="Matched" className="w-full h-full rounded-lg object-cover grayscale" />
                    </div>
                    <p className="mt-2 text-sm font-semibold text-green-400">Predicted Match</p>
                    <p className="text-xs text-slate-400">(Simulated Result)</p>
                </div>
            )}
        </div>
    );
}

export const InteractiveDemo: React.FC<InteractiveDemoProps> = ({ stepId }) => {
    const [image, setImage] = useState<string | null>(null);
    const [isImposter, setIsImposter] = useState<boolean>(false);
    const [showResult, setShowResult] = useState(false);
    const [isLoading, setIsLoading] = useState(false);

    const handleImageUpload = (file: File) => {
        const reader = new FileReader();
        reader.onloadend = () => {
            setImage(reader.result as string);
            setShowResult(false);
            setIsLoading(false);
        };
        reader.readAsDataURL(file);
    };
    
    const handleRecognition = () => {
        setIsLoading(true);
        setTimeout(() => {
            setShowResult(true);
            setIsLoading(false);
        }, 1500);
    }

    const renderContent = () => {
        if (!image) {
            return <UploadPlaceholder onImageUpload={handleImageUpload} />;
        }

        const imageStyle = stepId === 'test-2' ? 'grayscale contrast-125' : '';

        return (
            <div className="flex flex-col items-center gap-6">
                <div className="relative w-48 h-48">
                    <img src={image} alt="Test" className={`w-full h-full rounded-lg object-cover shadow-lg ${imageStyle}`} />
                    {stepId === 'test-3' && <div className="absolute inset-0 bg-black/50 flex items-center justify-center rounded-lg backdrop-blur-sm"><p className="text-white font-mono text-sm">[0.8, -0.2, ...]</p></div>}
                </div>

                {stepId === 'test-4' ? (
                     <div className="w-full max-w-md space-y-4">
                        {!showResult && (
                             <>
                                <div className="flex items-center justify-center gap-3 text-slate-300">
                                    <input
                                        type="checkbox"
                                        id="imposter-check"
                                        checked={isImposter}
                                        onChange={(e) => setIsImposter(e.target.checked)}
                                        className="h-4 w-4 rounded bg-slate-800 border-slate-600 text-sky-500 focus:ring-2 focus:ring-offset-0 focus:ring-offset-slate-900 focus:ring-sky-500 cursor-pointer"
                                    />
                                    <label htmlFor="imposter-check" className="text-sm font-medium cursor-pointer">Test with an imposter</label>
                                </div>
                                <button
                                    onClick={handleRecognition}
                                    disabled={isLoading}
                                    className="w-full bg-sky-600 text-white font-bold py-2 px-4 rounded-lg hover:bg-sky-500 transition-colors flex items-center justify-center disabled:bg-slate-600"
                                >
                                    {isLoading ? <><LoadingSpinner /> <span className="ml-2">Predicting...</span></> : 'Predict with ANN (Simulated)'}
                                </button>
                             </>
                        )}
                        {showResult && <ResultDisplay uploadedImage={image} isImposter={isImposter} />}
                    </div>
                ) : <p className="text-sm text-slate-400">Proceed to the next step to see the transformation.</p>}
            </div>
        );
    }

    return (
        <div className="bg-slate-900/50 p-6 rounded-lg border border-slate-700">
             <h4 className="text-lg font-semibold text-slate-300 mb-4">Interactive Testing Demo</h4>
             {renderContent()}
        </div>
    );
};