/**
 * Soil presets. Values are representative engineering figures, not site-measured:
 *  - swell: loose volume / bank volume after excavation (typical earthmoving tables: clay 1.25-1.40,
 *    common earth 1.20-1.30, sand/gravel 1.10-1.15)
 *  - reposeDeg: angle of repose of loose (dumped) material
 *  - bankDeg: steepest face the undisturbed (bank) soil holds before sloughing in this sim.
 *    Real stand-up time depends on cohesion, moisture and depth; this is a simplified critical angle.
 *  - density: bank density, t/m^3
 */
export interface SoilType {
  id: string;
  label: string;
  swell: number;
  reposeDeg: number;
  bankDeg: number;
  density: number;
  /** subsoil tint relative to the learned surface palette (multiplier) */
  subsoilTint: [number, number, number];
  /** loose spoil tint multiplier */
  spoilTint: [number, number, number];
  note: string;
}

export const SOILS: SoilType[] = [
  {
    id: 'wet-clay',
    label: 'Wet clay / mud',
    swell: 1.3,
    reposeDeg: 30,
    bankDeg: 62,
    density: 1.9,
    subsoilTint: [0.84, 0.82, 0.8],
    spoilTint: [1.0, 0.99, 0.98],
    note: 'Cohesive: cut faces stand steep, dumped spoil slumps low. Matches the muddy site in the capture.',
  },
  {
    id: 'common-earth',
    label: 'Common earth',
    swell: 1.25,
    reposeDeg: 34,
    bankDeg: 50,
    density: 1.8,
    subsoilTint: [0.9, 0.82, 0.72],
    spoilTint: [1.02, 0.97, 0.9],
    note: 'Mixed soil, moderate cohesion.',
  },
  {
    id: 'sandy-gravel',
    label: 'Sandy gravel',
    swell: 1.12,
    reposeDeg: 36,
    bankDeg: 40,
    density: 1.95,
    subsoilTint: [1.05, 0.98, 0.86],
    spoilTint: [1.1, 1.04, 0.94],
    note: 'Granular, little cohesion: trench walls slough toward the repose angle.',
  },
  {
    id: 'saturated-silt',
    label: 'Saturated silt',
    swell: 1.2,
    reposeDeg: 22,
    bankDeg: 32,
    density: 1.85,
    subsoilTint: [0.62, 0.6, 0.58],
    spoilTint: [0.72, 0.7, 0.68],
    note: 'Unstable when wet: walls cave in easily, spoil spreads wide.',
  },
];

export const soilById = (id: string): SoilType => SOILS.find((s) => s.id === id) ?? SOILS[0];
