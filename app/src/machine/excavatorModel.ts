import * as THREE from 'three';
import { DX225_GEOMETRY, type JointState, type LinkageGeometry } from './kinematics';

/**
 * Articulated mesh twin of a 22 t excavator. Proportions follow the DX225LC-7 spec sheet (track length,
 * gauge, shoe width, house width, cab height, boom/arm lengths). It is deliberately a clean engineering
 * model rather than a photoreal asset: the hybrid "splat world + mesh machine" split is the point.
 */
export class ExcavatorModel {
  readonly root = new THREE.Group();
  readonly under = new THREE.Group();
  readonly upper = new THREE.Group();
  readonly boomPivot = new THREE.Group();
  readonly armPivot = new THREE.Group();
  readonly bucketPivot = new THREE.Group();
  /** sensor mounts (children of the upper structure) */
  readonly lidarMount = new THREE.Object3D();
  readonly cameraMount = new THREE.Object3D();
  /** forward lidar tilted down for near-field grade inspection */
  readonly frontLidarMount = new THREE.Object3D();
  readonly g: LinkageGeometry;
  joints: JointState = { swing: 0, boom: 0.3, arm: -1.6, bucket: -1.2 };
  private cyl: { mesh: THREE.Mesh; rod: THREE.Mesh; a: THREE.Object3D; b: THREE.Object3D }[] = [];
  readonly bucketTip = new THREE.Object3D();
  private beacon: THREE.Mesh;

  constructor(g: LinkageGeometry = DX225_GEOMETRY) {
    this.g = g;
    const paint = new THREE.MeshStandardMaterial({ color: 0xe8701e, roughness: 0.5, metalness: 0.15 });
    const dark = new THREE.MeshStandardMaterial({ color: 0x24221f, roughness: 0.85, metalness: 0.2 });
    const steel = new THREE.MeshStandardMaterial({ color: 0x8d9298, roughness: 0.35, metalness: 0.7 });
    const glass = new THREE.MeshStandardMaterial({ color: 0x1b2730, roughness: 0.1, metalness: 0.4, transparent: true, opacity: 0.72 });
    const grey = new THREE.MeshStandardMaterial({ color: 0x55534f, roughness: 0.7, metalness: 0.25 });

    const box = (w: number, h: number, d: number, m: THREE.Material) => { const o = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), m); o.castShadow = true; return o; };

    // ---------- undercarriage (local +x = track direction) ----------
    const trackLen = 4.445, gauge = 2.39, shoe = 0.6, trackH = 0.945;
    for (const side of [-1, 1]) {
      const t = new THREE.Group();
      const frame = box(trackLen - 0.9, trackH * 0.55, shoe * 0.7, grey);
      frame.position.y = trackH * 0.5;
      t.add(frame);
      // track loop as a capsule-ish shape: straight runs + end drums
      const run = new THREE.Mesh(new THREE.CapsuleGeometry(trackH / 2, trackLen - trackH, 6, 16), dark);
      run.rotation.z = Math.PI / 2;
      run.scale.set(1, 1, shoe / trackH);
      run.position.y = trackH / 2;
      t.add(run);
      // shoes (grousers): instanced thin bars on top and bottom runs
      const nShoe = 44;
      const inst = new THREE.InstancedMesh(new THREE.BoxGeometry(0.07, 0.035, shoe * 1.02), steel, nShoe);
      const m = new THREE.Matrix4();
      for (let i = 0; i < nShoe; i++) {
        const half = nShoe / 2;
        const top = i < half;
        const x = -trackLen / 2 + trackH / 2 + ((i % half) / (half - 1)) * (trackLen - trackH);
        m.makeTranslation(x, top ? trackH + 0.015 : -0.015, 0);
        inst.setMatrixAt(i, m);
      }
      t.add(inst);
      for (const ex of [-1, 1]) {
        const drum = new THREE.Mesh(new THREE.CylinderGeometry(trackH * 0.36, trackH * 0.36, shoe * 1.05, 20), steel);
        drum.rotation.x = Math.PI / 2;
        drum.position.set(ex * (trackLen / 2 - trackH / 2), trackH / 2, 0);
        t.add(drum);
      }
      t.position.z = (side * gauge) / 2;
      this.under.add(t);
    }
    const carbody = box(1.8, 0.55, gauge - shoe * 0.6, grey);
    carbody.position.y = 0.75;
    this.under.add(carbody);
    const ring = new THREE.Mesh(new THREE.CylinderGeometry(0.95, 0.95, 0.12, 28), steel);
    ring.position.y = 1.05;
    this.under.add(ring);

    // ---------- upper structure (local +x = forward / boom heading) ----------
    const clearance = 1.092, houseW = 2.71, houseH = 0.95;
    const deck = box(3.1, 0.18, houseW, grey);
    deck.position.set(-0.55, clearance + 0.09, 0);
    this.upper.add(deck);
    const engine = box(1.75, houseH, houseW - 0.1, paint);
    engine.position.set(-1.35, clearance + 0.18 + houseH / 2, 0);
    this.upper.add(engine);
    const cw = new THREE.Mesh(new THREE.CylinderGeometry(2.0, 2.0, 1.05, 32, 1, false, Math.PI * 0.62, Math.PI * 0.76), paint);
    cw.position.set(0.0, clearance + 0.55, 0);
    cw.rotation.y = 0;
    this.upper.add(cw);
    const cwFace = box(0.5, 1.05, houseW, paint);
    cwFace.position.set(-2.2, clearance + 0.55, 0);
    this.upper.add(cwFace);
    // right-side tool box / tank
    const tank = box(1.1, 0.85, 0.9, paint);
    tank.position.set(0.6, clearance + 0.18 + 0.425, 0.85);
    this.upper.add(tank);
    // cab: front-left (left = +z in this frame when looking along +x? three is right-handed: +z is to the right of +x when looking down -y)
    const cabW = 1.01, cabH = 2.98 - clearance - 0.18, cabL = 1.75;
    const cab = new THREE.Group();
    const cabShell = box(cabL, cabH, cabW, paint);
    cabShell.position.y = cabH / 2;
    cab.add(cabShell);
    const winF = box(0.02, cabH * 0.62, cabW * 0.86, glass); winF.position.set(cabL / 2 + 0.011, cabH * 0.58, 0); cab.add(winF);
    for (const s of [-1, 1]) { const winS = box(cabL * 0.8, cabH * 0.55, 0.02, glass); winS.position.set(0.05, cabH * 0.6, s * (cabW / 2 + 0.011)); cab.add(winS); }
    cab.position.set(0.55, clearance + 0.18, -(houseW / 2 - cabW / 2 - 0.05));
    this.upper.add(cab);
    // handrails
    const railM = new THREE.MeshStandardMaterial({ color: 0xd6d9db, roughness: 0.4, metalness: 0.6 });
    const railY = clearance + 0.18 + houseH + 0.42;
    for (const z of [-houseW / 2 + 0.05, -0.2]) { const r = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 1.7, 6), railM); r.rotation.z = Math.PI / 2; r.position.set(-1.35, railY, z); this.upper.add(r); }
    this.beacon = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.08, 0.14, 12), new THREE.MeshStandardMaterial({ color: 0xffa31a, emissive: 0xff7a00, emissiveIntensity: 0.6 }));
    this.beacon.position.set(0.3, clearance + 0.18 + cabH + 0.08, -(houseW / 2 - cabW / 2));
    this.upper.add(this.beacon);

    // sensors: roof lidar puck above the cab, forward camera at the cab roof's front edge
    const puck = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.045, 0.075, 18), new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.3, metalness: 0.5 }));
    const puckBase = box(0.18, 0.06, 0.18, dark);
    const lidarY = clearance + 0.18 + cabH + 0.2;
    this.lidarMount.position.set(0.35, lidarY, -(houseW / 2 - cabW / 2));
    puck.position.copy(this.lidarMount.position);
    puckBase.position.copy(this.lidarMount.position).add(new THREE.Vector3(0, -0.07, 0));
    this.upper.add(puck, puckBase, this.lidarMount);
    // forward camera on a short mast above the cab's front edge (clear of the cab/boom geometry)
    this.cameraMount.position.set(0.55 + cabL / 2 + 0.45, clearance + 0.18 + cabH + 0.55, -(houseW / 2 - cabW / 2) - 0.1);
    const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, 0.55, 8), steel);
    mast.position.set(0.55 + cabL / 2 + 0.2, clearance + 0.18 + cabH + 0.27, -(houseW / 2 - cabW / 2) - 0.1);
    const camBox = box(0.14, 0.1, 0.12, dark);
    camBox.position.copy(this.cameraMount.position);
    this.upper.add(mast, camBox);
    this.frontLidarMount.position.set(0.55 + cabL / 2 + 0.12, clearance + 0.18 + cabH + 0.12, -(houseW / 2 - cabW / 2) + 0.28);
    const fl = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.045, 0.075, 18), new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.3, metalness: 0.5 }));
    fl.position.copy(this.frontLidarMount.position);
    fl.rotation.z = -35 * Math.PI / 180;
    this.upper.add(this.frontLidarMount, fl);
    this.upper.add(this.cameraMount);

    // ---------- front linkage ----------
    this.boomPivot.position.set(g.boomFootR, g.boomFootY, 0.05);
    this.upper.add(this.boomPivot);
    // boom: banana shape from two tapered segments (kinematic line is straight foot->elbow)
    const kinkF = 0.42, kinkUp = 0.55;
    const kink = new THREE.Vector3(g.boom * kinkF, kinkUp, 0);
    const seg = (from: THREE.Vector3, to: THREE.Vector3, w0: number, w1: number, depth: number) => {
      const len = from.distanceTo(to);
      const geo = new THREE.BoxGeometry(len, 1, depth);
      // taper height along x
      const pos = geo.attributes.position as THREE.BufferAttribute;
      for (let i = 0; i < pos.count; i++) { const x = pos.getX(i) / len + 0.5; pos.setY(i, pos.getY(i) * (w0 + (w1 - w0) * x)); }
      geo.computeVertexNormals();
      const mesh = new THREE.Mesh(geo, paint);
      mesh.castShadow = true;
      mesh.position.copy(from).add(to).multiplyScalar(0.5);
      mesh.rotation.z = Math.atan2(to.y - from.y, to.x - from.x);
      return mesh;
    };
    this.boomPivot.add(seg(new THREE.Vector3(0, 0, 0), kink, 0.62, 0.78, 0.62));
    this.boomPivot.add(seg(kink, new THREE.Vector3(g.boom, 0, 0), 0.78, 0.48, 0.55));
    this.armPivot.position.set(g.boom, 0, 0);
    this.boomPivot.add(this.armPivot);
    const armMesh = seg(new THREE.Vector3(-0.45, 0.15, 0), new THREE.Vector3(g.arm, 0, 0), 0.62, 0.32, 0.42);
    this.armPivot.add(armMesh);
    this.bucketPivot.position.set(g.arm, 0, 0);
    this.armPivot.add(this.bucketPivot);
    // bucket: extruded profile (pin at origin, tip at +x)
    const shape = new THREE.Shape();
    shape.moveTo(0, 0.12);
    shape.quadraticCurveTo(0.35, 0.45, 0.85, 0.42);
    shape.lineTo(g.bucket, 0.06);
    shape.lineTo(g.bucket - 0.05, -0.05);
    shape.quadraticCurveTo(0.7, -0.62, 0.15, -0.35);
    shape.lineTo(0, 0.12);
    const bw = 1.05;
    const bgeo = new THREE.ExtrudeGeometry(shape, { depth: bw, bevelEnabled: false });
    bgeo.translate(0, 0, -bw / 2);
    const bucket = new THREE.Mesh(bgeo, grey);
    bucket.castShadow = true;
    this.bucketPivot.add(bucket);
    const teeth = new THREE.InstancedMesh(new THREE.ConeGeometry(0.045, 0.16, 6), steel, 5);
    const mt = new THREE.Matrix4();
    for (let i = 0; i < 5; i++) { mt.makeRotationZ(-Math.PI / 2 - 0.5); mt.setPosition(g.bucket + 0.03, 0.02, -bw / 2 + 0.11 + (i * (bw - 0.22)) / 4); teeth.setMatrixAt(i, mt); }
    this.bucketPivot.add(teeth);
    this.bucketTip.position.set(g.bucket, 0, 0);
    this.bucketPivot.add(this.bucketTip);

    // hydraulic cylinders (barrel + rod) between anchor points that move with the joints
    const anchor = (parent: THREE.Object3D, x: number, y: number) => { const o = new THREE.Object3D(); o.position.set(x, y, 0); parent.add(o); return o; };
    const addCyl = (a: THREE.Object3D, b: THREE.Object3D, r: number) => {
      const mesh = new THREE.Mesh(new THREE.CylinderGeometry(r, r, 1, 14), dark);
      const rod = new THREE.Mesh(new THREE.CylinderGeometry(r * 0.55, r * 0.55, 1, 10), railM);
      this.root.add(mesh, rod);
      this.cyl.push({ mesh, rod, a, b });
    };
    addCyl(anchor(this.upper, 1.25, clearance + 0.3), anchor(this.boomPivot, 2.3, -0.25), 0.09);
    addCyl(anchor(this.boomPivot, 2.6, 0.95), anchor(this.armPivot, -0.42, 0.42), 0.085);
    addCyl(anchor(this.armPivot, 0.55, 0.36), anchor(this.armPivot, g.arm - 0.25, 0.32), 0.07);

    this.root.add(this.under, this.upper);
    this.root.traverse((o) => { if ((o as THREE.Mesh).isMesh) (o as THREE.Mesh).castShadow = true; });
  }

  /** place at the machine's swing centre; yaw angles use atan2(z, x) convention */
  place(cx: number, groundY: number, cz: number, trackYaw: number) {
    this.root.position.set(cx, groundY, cz);
    this.under.rotation.y = -trackYaw;
  }

  setJoints(j: JointState) {
    this.joints = j;
    this.upper.rotation.y = -j.swing;
    this.boomPivot.rotation.z = j.boom;
    this.armPivot.rotation.z = j.arm;
    this.bucketPivot.rotation.z = j.bucket;
    this.root.updateMatrixWorld(true);
    const pa = new THREE.Vector3(), pb = new THREE.Vector3(), dir = new THREE.Vector3(), up = new THREE.Vector3(0, 1, 0), q = new THREE.Quaternion();
    const inv = new THREE.Matrix4().copy(this.root.matrixWorld).invert();
    for (const c of this.cyl) {
      c.a.getWorldPosition(pa).applyMatrix4(inv);
      c.b.getWorldPosition(pb).applyMatrix4(inv);
      dir.subVectors(pb, pa);
      const len = dir.length();
      dir.normalize();
      q.setFromUnitVectors(up, dir);
      const barrel = len * 0.58;
      c.mesh.scale.set(1, barrel, 1);
      c.mesh.quaternion.copy(q);
      c.mesh.position.copy(pa).addScaledVector(dir, barrel / 2);
      const rodLen = len * 0.5;
      c.rod.scale.set(1, rodLen, 1);
      c.rod.quaternion.copy(q);
      c.rod.position.copy(pb).addScaledVector(dir, -rodLen / 2);
    }
  }

  tick(t: number) {
    const m = this.beacon.material as THREE.MeshStandardMaterial;
    m.emissiveIntensity = 0.35 + 0.65 * Math.max(0, Math.sin(t * 6));
  }

  tipWorld(out = new THREE.Vector3()) {
    return this.bucketTip.getWorldPosition(out);
  }
}
