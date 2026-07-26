"use client"

import * as React from "react"
import { Separator as SeparatorPrimitive } from "radix-ui"

import { cn } from "@/lib/utils"

function Separator({
  className,
  orientation = "horizontal",
  decorative = true,
  ...props
}: React.ComponentProps<typeof SeparatorPrimitive.Root>) {
  return (
    <SeparatorPrimitive.Root
      data-slot="separator"
      decorative={decorative}
      orientation={orientation}
      className={cn(
        "shrink-0 border-border",
        // Use border (not bg + h-px) so both ends render the same 1px on all DPIs.
        orientation === "horizontal"
          ? "h-0 w-full border-t"
          : "h-full w-0 self-stretch border-l",
        className
      )}
      {...props}
    />
  )
}

export { Separator }
