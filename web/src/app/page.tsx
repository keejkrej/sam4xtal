import { SemWorkspace } from "@/components/sem/sem-workspace";
import { TooltipProvider } from "@/components/ui/tooltip";

export default function HomePage() {
  return (
    <TooltipProvider>
      <main className="min-h-full flex-1 bg-background">
        <SemWorkspace />
      </main>
    </TooltipProvider>
  );
}
