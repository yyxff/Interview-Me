import { useRef, useEffect, useState } from 'react';

export type CameraPermission = 'pending' | 'granted' | 'denied';

export function useWebcam() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [permission, setPermission] = useState<CameraPermission>('pending');

  useEffect(() => {
    let cancelled = false;

    navigator.mediaDevices
      .getUserMedia({ video: true, audio: false })
      .then((stream) => {
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        setPermission('granted');
        return () => stream.getTracks().forEach((t) => t.stop());
      })
      .catch(() => {
        if (!cancelled) setPermission('denied');
      });

    return () => {
      cancelled = true;
      if (videoRef.current?.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  return { videoRef, permission };
}
