
import { GoogleGenAI } from "@google/genai";
import { Step } from '../types';

if (!process.env.API_KEY) {
  throw new Error("API_KEY environment variable is not set");
}

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

export async function getExplanationForStep(step: Step): Promise<string> {
  const prompt = `
    You are an expert in machine learning and computer vision, tasked with explaining a concept to a talented frontend engineer.
    Your tone should be clear, intuitive, and engaging.

    Explain the following step in the PCA/Eigenfaces algorithm for face recognition. Use Markdown for formatting (headings, lists, bold text, italics).
    Avoid overly complex mathematical formulas and focus on the 'why' behind the step. Use an analogy if it helps clarify the concept.

    **Step Title: ${step.title}**

    **Step Details from Document:**
    > ${step.description}

    **Your explanation should cover:**
    1.  **Goal:** What is the main objective of this step?
    2.  **Intuition/Analogy:** Explain it in simple terms. For example, liken it to finding the most important ingredients in a recipe.
    3.  **Significance:** Why is this step crucial for the overall process of face recognition?
    `;
  
  try {
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: prompt,
    });
    return response.text;
  } catch (error) {
    console.error("Gemini API call failed:", error);
    return "Error: Could not generate an explanation. Please check the console for details.";
  }
}
