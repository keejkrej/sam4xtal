import { Workspace } from "@/components/workspace/workspace";
import { TooltipProvider } from "@/components/ui/tooltip";

export default function HomePage() {
  return (
    <TooltipProvider>
      <main className="flex min-h-0 flex-1 flex-col bg-background">
        <Workspace />
      </main>
    </TooltipProvider>
  );
}
