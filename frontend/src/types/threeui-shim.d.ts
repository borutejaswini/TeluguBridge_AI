declare module '@designcodeio/threeui' {
  import type { ComponentType } from 'react'

  export interface ElementsCollectionProps {
    variant: 'condensation' | string
    speed?: number
    dropAmount?: number
    opacity?: number
  }
  export const ElementsCollection: ComponentType<ElementsCollectionProps>

  export interface GlobeCollectionProps {
    variant: 'energy-orb' | string
    speed?: number
    scale?: number
    smokeScale?: number
    smokeStrength?: number
    smokeSpeed?: number
    hue?: number
    saturation?: number
    glow?: number
    starDensity?: number
    starSpeed?: number
    starSize?: number
    brightness?: number
    opacity?: number
  }
  export const GlobeCollection: ComponentType<GlobeCollectionProps>
}

declare module '@designcodeio/threeui/style.css'
