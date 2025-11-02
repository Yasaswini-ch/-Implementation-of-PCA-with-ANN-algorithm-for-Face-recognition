
import React from 'react';

interface ImageGridProps {
  title: string;
  count: number;
  type: 'normal' | 'mean' | 'mean-zero' | 'eigenface';
}

const getFilterStyle = (type: 'normal' | 'mean' | 'mean-zero' | 'eigenface') => {
    switch (type) {
        case 'normal':
            return 'grayscale';
        case 'mean':
            return 'grayscale blur-sm';
        case 'mean-zero':
            return 'grayscale contrast-125';
        case 'eigenface':
            return 'grayscale contrast-200 invert';
        default:
            return 'grayscale';
    }
}

export const ImageGrid: React.FC<ImageGridProps> = ({ title, count, type }) => {
  const images = Array.from({ length: count }, (_, i) => `https://picsum.photos/seed/${type}${i}/200/200`);
  const filterStyle = getFilterStyle(type);

  return (
    <div>
      <h4 className="text-lg font-semibold text-slate-300 mb-4">{title}</h4>
      <div className={`grid gap-4 ${count === 1 ? 'max-w-xs mx-auto' : 'grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6'}`}>
        {images.map((src, index) => (
          <div key={index} className="aspect-square bg-slate-700 rounded-md overflow-hidden shadow-md">
            <img 
              src={src} 
              alt={`Face ${index + 1}`} 
              className={`w-full h-full object-cover transition-transform duration-300 hover:scale-105 ${filterStyle}`}
            />
          </div>
        ))}
      </div>
    </div>
  );
};
