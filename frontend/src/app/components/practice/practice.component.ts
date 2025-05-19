import { Component, OnInit, ViewChild, ElementRef } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { HttpClientModule } from '@angular/common/http';

@Component({
  selector: 'app-practice',
  standalone: true,
  templateUrl: './practice.component.html',
  styleUrls: ['./practice.component.css'],
  imports: [CommonModule, HttpClientModule]
})
export class PracticeComponent implements OnInit {
  detectedText: string = '';
  words: string[] = [];
  isDetecting: boolean = true;
  hideImage = false;
  

  tips: string[] = [
    'Coloca tu mano centrada frente a la cámara 📷',
    'Evita fondos con ruido visual 🎨',
    'Realiza el gesto con claridad y sin moverte 🖐️',
    'Utiliza buena iluminación 💡',
    'Haz pausas entre gestos para mejor precisión ⏸️'
  ];
  currentTip: string = this.tips[0];
  private tipIndex: number = 0;

  @ViewChild('video', { static: true }) videoElement!: ElementRef<HTMLVideoElement>;

  constructor(private http: HttpClient) {}

  ngOnInit() {
    this.setupCamera();

    // Actualiza el texto detectado cada segundo
    setInterval(() => {
      this.http.get<{ current_text: string }>('http://localhost:5001/get_text')
        .subscribe(response => {
          const text = response.current_text.trim();
          this.detectedText = text.charAt(0).toUpperCase() + text.slice(1);
          this.words = this.detectedText.split(' ').filter(w => w !== '');
        });
    }, 1000);

    // Cambiar tip cada 5 segundos
    setInterval(() => {
      this.tipIndex = (this.tipIndex + 1) % this.tips.length;
      this.currentTip = this.tips[this.tipIndex];
    }, 5000);
  }

  clearAll() {
    this.words = [];
  
    this.clearText(); // Llama también a tu función para borrar del backend
  }
  
  clearText() {
    this.http.post('http://localhost:5001/clear', {}).subscribe(() => {
      this.detectedText = '';
      this.words = [];
    });
  }

  setupCamera() {
    navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
      this.videoElement.nativeElement.srcObject = stream;
    }).catch(error => {
      console.error('Error al acceder a la cámara:', error);
    });
  }
}
