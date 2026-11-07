import * as THREE from 'three';
import { PointLabel, type Capsule } from '../sensors/lidarModel';

/** a ground worker in hi-vis PPE; also exposes a capsule for the lidar */
export class Worker {
  readonly group = new THREE.Group();
  x = 0; z = 0; y = 0;
  constructor() {
    const vest = new THREE.MeshStandardMaterial({ color: 0xd8ff1f, emissive: 0x3a4a00, roughness: 0.6 });
    const pants = new THREE.MeshStandardMaterial({ color: 0x1f2a3a, roughness: 0.8 });
    const skin = new THREE.MeshStandardMaterial({ color: 0xc9a07a, roughness: 0.7 });
    const hat = new THREE.MeshStandardMaterial({ color: 0xf2f2ee, roughness: 0.35 });
    const stripe = new THREE.MeshStandardMaterial({ color: 0xe8e8e8, emissive: 0x555555, roughness: 0.2, metalness: 0.4 });
    for (const s of [-1, 1]) { const leg = new THREE.Mesh(new THREE.CapsuleGeometry(0.075, 0.72, 4, 10), pants); leg.position.set(0, 0.45, s * 0.11); this.group.add(leg); }
    const torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.19, 0.42, 6, 14), vest); torso.position.y = 1.13; this.group.add(torso);
    const band = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.2, 0.05, 16), stripe); band.position.y = 1.08; this.group.add(band);
    for (const s of [-1, 1]) { const arm = new THREE.Mesh(new THREE.CapsuleGeometry(0.06, 0.5, 4, 8), vest); arm.position.set(0.02, 1.12, s * 0.27); arm.rotation.x = s * 0.12; this.group.add(arm); }
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.11, 16, 12), skin); head.position.y = 1.56; this.group.add(head);
    const helmet = new THREE.Mesh(new THREE.SphereGeometry(0.135, 18, 10, 0, Math.PI * 2, 0, Math.PI / 2), hat); helmet.position.y = 1.6; this.group.add(helmet);
    this.group.traverse((o) => { if ((o as THREE.Mesh).isMesh) (o as THREE.Mesh).castShadow = true; });
    this.group.visible = false;
  }
  place(x: number, y: number, z: number, faceYaw = 0) {
    this.x = x; this.y = y; this.z = z;
    this.group.position.set(x, y, z);
    this.group.rotation.y = -faceYaw;
  }
  capsule(): Capsule { return { x: this.x, z: this.z, y0: this.y + 0.25, y1: this.y + 1.55, r: 0.26, label: PointLabel.Person }; }
}
