import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';

const ThreeBackground = () => {
  const mountRef = useRef(null);

  useEffect(() => {
    if (!mountRef.current) return;

    // Scene, camera, renderer
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    const renderer = new THREE.WebGLRenderer({ alpha: false, antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    
    // Set clear color to pure black
    renderer.setClearColor(0x000000, 1);

    mountRef.current.appendChild(renderer.domElement);

    // Create very minimal starfield
    const starsGeometry = new THREE.BufferGeometry();
    const starsCount = 200;
    const starsPositions = new Float32Array(starsCount * 3);

    for (let i = 0; i < starsCount; i++) {
      starsPositions[i * 3] = (Math.random() - 0.5) * 300;
      starsPositions[i * 3 + 1] = (Math.random() - 0.5) * 300;
      starsPositions[i * 3 + 2] = (Math.random() - 0.5) * 300;
    }

    starsGeometry.setAttribute('position', new THREE.BufferAttribute(starsPositions, 3));

    const starsMaterial = new THREE.PointsMaterial({
      size: 1,
      color: 0xffffff,
      transparent: true,
      opacity: 0.4
    });

    const starsMesh = new THREE.Points(starsGeometry, starsMaterial);
    scene.add(starsMesh);

    // Create occasional meteors
    class Meteor {
      constructor() {
        this.geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(6);
        this.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        
        this.material = new THREE.LineBasicMaterial({
          color: new THREE.Color(0.95, 0.95, 1.0),
          transparent: true,
          opacity: 0
        });
        
        this.mesh = new THREE.Line(this.geometry, this.material);
        this.reset();
        scene.add(this.mesh);
      }
      
      reset() {
        const startX = -60 + Math.random() * 40;
        const startY = 40 + Math.random() * 20;
        const startZ = -30 + Math.random() * 20;
        
        this.startPos = new THREE.Vector3(startX, startY, startZ);
        
        this.direction = new THREE.Vector3(
          0.8 + Math.random() * 0.2,
          -0.9 - Math.random() * 0.3,
          0.1
        ).normalize();
        
        this.speed = 0.1 + Math.random() * 0.05;
        this.length = 6 + Math.random() * 4;
        this.progress = 0;
        this.maxProgress = 12 + Math.random() * 8;
        this.material.opacity = 0;
        this.delay = Math.random() * 15 + 5; // Much longer delays
      }
      
      update() {
        if (this.delay > 0) {
          this.delay -= 0.016;
          return;
        }
        
        this.progress += 0.012;
        
        if (this.progress > this.maxProgress) {
          this.reset();
          return;
        }
        
        // Very subtle fade
        if (this.progress < 2) {
          this.material.opacity = (this.progress / 2) * 0.6;
        } else if (this.progress > this.maxProgress - 3) {
          this.material.opacity = ((this.maxProgress - this.progress) / 3) * 0.6;
        } else {
          this.material.opacity = 0.6;
        }
        
        const currentPos = this.startPos.clone().add(
          this.direction.clone().multiplyScalar(this.progress * this.speed * 20)
        );
        
        const tailPos = currentPos.clone().add(
          this.direction.clone().multiplyScalar(-this.length)
        );
        
        const positions = this.geometry.attributes.position.array;
        positions[0] = tailPos.x;
        positions[1] = tailPos.y;
        positions[2] = tailPos.z;
        positions[3] = currentPos.x;
        positions[4] = currentPos.y;
        positions[5] = currentPos.z;
        
        this.geometry.attributes.position.needsUpdate = true;
      }
    }

    // Create only 2 meteors for subtlety
    const meteors = [];
    for (let i = 0; i < 2; i++) {
      const meteor = new Meteor();
      meteor.delay = i * 10;
      meteors.push(meteor);
    }

    camera.position.z = 5;

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);

      // Barely visible star movement
      starsMesh.rotation.y += 0.00002;

      // Update meteors
      meteors.forEach(meteor => meteor.update());

      renderer.render(scene, camera);
    };
    animate();

    // Handle resize
    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };
    
    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => {
      const currentMount = mountRef.current;
      if (currentMount && renderer.domElement) {
        currentMount.removeChild(renderer.domElement);
      }
      window.removeEventListener('resize', handleResize);
      
      meteors.forEach(meteor => {
        scene.remove(meteor.mesh);
        meteor.geometry.dispose();
        meteor.material.dispose();
      });
      
      scene.clear();
      renderer.dispose();
    };
  }, []);

  return (
    <div 
      ref={mountRef} 
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        zIndex: -1,
        pointerEvents: 'none'
      }}
    />
  );
};

export default ThreeBackground;
