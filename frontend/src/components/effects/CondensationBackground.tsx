import { ElementsCollection } from '@designcodeio/threeui'
import '@designcodeio/threeui/style.css'
import { useReducedMotion } from '../../hooks/useReducedMotion'

// Official ThreeUI adapter. Renders the authored ElementsCollection "condensation"
// variant unmodified — no shader/renderer logic lives in this repo. Props are
// tuned once here (low opacity, slow speed) so it stays a subtle atmospheric
// layer rather than a dominant effect; see the master spec section 7A.
interface CondensationBackgroundProps {
  speed?: number
  dropAmount?: number
  opacity?: number
}

export default function CondensationBackground({ speed = 0.6, dropAmount = 0.65, opacity = 0.34 }: CondensationBackgroundProps) {
  const reducedMotion = useReducedMotion()

  // Reduced motion: stop the effect entirely rather than approximate a static frame,
  // since we must not alter the authored renderer's internals.
  if (reducedMotion) return null

  return (
    <div className="condensation-canvas" aria-hidden="true">
      <ElementsCollection variant="condensation" speed={speed} dropAmount={dropAmount} opacity={opacity} />
    </div>
  )
}
