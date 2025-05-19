// learn.component.ts
import { Component, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-learn',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './learn.component.html',
  styleUrls: ['./learn.component.css']
})
export class LearnComponent implements AfterViewInit {
  letters = [
    { char: 'A', imageUrl: 'assets/signs/A.png' },
    { char: 'B', imageUrl: 'assets/signs/B.png' },
    { char: 'C', imageUrl: 'assets/signs/C.png' },
    { char: 'D', imageUrl: 'assets/signs/D.png' },
    { char: 'E', imageUrl: 'assets/signs/E.png' },
    { char: 'F', imageUrl: 'assets/signs/F.png' },
    { char: 'G', imageUrl: 'assets/signs/G.png' },
    { char: 'H', imageUrl: 'assets/signs/H.png' },
    { char: 'I', imageUrl: 'assets/signs/I.png' },
    { char: 'J', imageUrl: 'assets/signs/J.png' },
    { char: 'K', imageUrl: 'assets/signs/K.png' },
    { char: 'L', imageUrl: 'assets/signs/L.png' },
    { char: 'M', imageUrl: 'assets/signs/M.png' },
    { char: 'N', imageUrl: 'assets/signs/N.png' },
    { char: 'Ñ', imageUrl: 'assets/signs/N-tilde.png' },
    { char: 'O', imageUrl: 'assets/signs/O.png' }
  ];

  ngAfterViewInit(): void {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
          observer.unobserve(entry.target);
        }
      });
    }, {
      threshold: 0.1
    });

    const elements = document.querySelectorAll('.fade-in-section');
    elements.forEach(el => observer.observe(el));
  }
}
