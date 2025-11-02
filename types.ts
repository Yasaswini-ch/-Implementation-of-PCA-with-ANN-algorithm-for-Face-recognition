
export interface Step {
  id: string;
  title: string;
  description: string;
  hasVisual: boolean;
  visualType?: 'dataset' | 'mean' | 'mean-zero' | 'eigenfaces' | 'interactive';
}
