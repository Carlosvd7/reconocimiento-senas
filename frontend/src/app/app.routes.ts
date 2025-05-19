import { Routes } from '@angular/router';
import { LearnComponent } from './components/learn/learn.component';
import { PracticeComponent } from './components/practice/practice.component';
import { GameComponent } from './components/game/game.component';


export const routes: Routes = [
  { path: '', component: LearnComponent },
  { path: 'practice', component: PracticeComponent },
  { path: 'game', component: GameComponent }
];






