import { LetterCard } from "./LetterCard";

interface LetterGridProps {
  letters: string[];
}

export const LetterGrid = ({ letters }: LetterGridProps) => {
  return (
    <div className="grid grid-cols-3 gap-3 sm:grid-cols-4 md:grid-cols-6">
      {letters.map((letter) => (
        <LetterCard key={letter} letter={letter} />
      ))}
    </div>
  );
};
