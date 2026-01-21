/*
 * GPU Learning - Interactive Components
 * Combined course interactivity + reusable components
 */

document.addEventListener('DOMContentLoaded', () => {
  initNav();
  initProgress();
  initSections();
  initHierarchy();
  initQuiz();
  initSliders();
  initSchedule();
  initChecklist();
  initMemBars();
});

// ============================================
// Navigation
// ============================================
function initNav() {
  // Mobile toggle
  document.querySelector('.mobile-btn')?.addEventListener('click', () => {
    document.querySelector('.sidebar')?.classList.toggle('open');
    document.querySelector('nav')?.classList.toggle('open');
  });
  
  // Section toggles (nav accordion)
  document.querySelectorAll('.nav-section__btn, .nav-section-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      btn.closest('.nav-section').classList.toggle('open');
    });
  });
  
  // Active tracking via intersection observer
  const sections = document.querySelectorAll('.section[id]');
  if (sections.length > 0) {
    const observer = new IntersectionObserver(entries => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          document.querySelectorAll('.nav-item').forEach(a => {
            a.classList.toggle('active', a.getAttribute('href') === '#' + e.target.id);
          });
        }
      });
    }, { rootMargin: '-20% 0px -70% 0px' });
    
    sections.forEach(s => observer.observe(s));
  }
}

// ============================================
// Progress Bar
// ============================================
function initProgress() {
  const bar = document.getElementById('progress-bar');
  if (!bar) return;
  
  const update = () => {
    const h = document.documentElement.scrollHeight - window.innerHeight;
    bar.style.width = h > 0 ? (window.scrollY / h * 100) + '%' : '0%';
  };
  
  window.addEventListener('scroll', update, { passive: true });
  update();
}

// ============================================
// Section Fade-in
// ============================================
function initSections() {
  const observer = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) e.target.classList.add('visible');
    });
  }, { rootMargin: '0px 0px -10% 0px', threshold: 0.1 });
  
  document.querySelectorAll('.section').forEach((s, i) => {
    observer.observe(s);
    if (i < 2) s.classList.add('visible'); // Show first 2 immediately
  });
}

// ============================================
// Hierarchy Interactive
// ============================================
function initHierarchy() {
  document.querySelectorAll('.hier-item').forEach(item => {
    item.addEventListener('click', () => {
      const id = item.dataset.detail;
      const wrap = item.closest('.hierarchy');
      
      wrap.querySelectorAll('.hier-item').forEach(i => i.classList.remove('active'));
      wrap.querySelectorAll('.hier-detail').forEach(d => d.classList.remove('show'));
      
      item.classList.add('active');
      document.getElementById(id)?.classList.add('show');
    });
  });
}

// ============================================
// Quiz
// ============================================
function initQuiz() {
  document.querySelectorAll('.quiz').forEach(quiz => {
    const opts = quiz.querySelectorAll('.quiz-opt, .quiz__option');
    const fb = quiz.querySelector('.quiz-fb, .quiz__feedback');
    
    opts.forEach(opt => {
      opt.addEventListener('click', () => {
        const correct = opt.dataset.correct === 'true';
        
        opts.forEach(o => o.classList.remove('picked', 'correct', 'wrong', 
          'quiz__option--selected', 'quiz__option--correct', 'quiz__option--incorrect'));
        
        opt.classList.add('picked', correct ? 'correct' : 'wrong');
        
        // Also add scaffold-style classes
        opt.classList.add('quiz__option--selected');
        opt.classList.add(correct ? 'quiz__option--correct' : 'quiz__option--incorrect');
        
        if (fb) {
          fb.className = fb.className.includes('quiz__feedback') 
            ? 'quiz__feedback show ' + (correct ? 'quiz__feedback--correct' : 'quiz__feedback--incorrect')
            : 'quiz-fb show ' + (correct ? 'ok' : 'no');
          fb.textContent = correct ? '✓ Correct!' : '✗ Not quite. Review the section above.';
        }
      });
    });
  });
}

// ============================================
// Sliders
// ============================================
function initSliders() {
  document.querySelectorAll('input[type="range"]').forEach(slider => {
    const valEl = document.getElementById(slider.dataset.target);
    const cb = slider.dataset.callback;
    
    const update = () => {
      if (valEl) valEl.textContent = slider.value;
      if (cb && window[cb]) window[cb](slider.value);
    };
    
    slider.addEventListener('input', update);
    update();
  });
}

// Pipeline tradeoff callback
window.updatePipeline = (n) => {
  const data = {
    1: { t: '1 Stage (No Pipelining)', p: 'Simplest, minimal SMEM', c: 'No latency hiding' },
    2: { t: '2 Stages', p: 'Lower SMEM, simpler sync', c: 'May not fully hide latency' },
    3: { t: '3 Stages', p: 'Good balance', c: 'Moderate SMEM overhead' },
    4: { t: '4 Stages', p: 'Excellent latency hiding', c: 'Higher SMEM, complex sync' },
    5: { t: '5+ Stages', p: 'Maximum hiding', c: 'Diminishing returns' }
  };
  const d = data[n] || data[2];
  const el = document.getElementById('pipe-info');
  if (el) el.innerHTML = `<h4>${d.t}</h4><p><strong class="text-green">Pros:</strong> ${d.p}</p><p class="mb-0"><strong class="text-red">Cons:</strong> ${d.c}</p>`;
};

// ============================================
// Schedule Days
// ============================================
function initSchedule() {
  document.querySelectorAll('.sched-head').forEach(head => {
    head.addEventListener('click', () => {
      head.closest('.sched-day').classList.toggle('open');
    });
  });
}

// ============================================
// Checklist
// ============================================
function initChecklist() {
  document.querySelectorAll('.check-box').forEach(box => {
    box.addEventListener('click', () => {
      box.classList.toggle('done');
      box.closest('.check-item').classList.toggle('done');
      
      // Check completion
      const list = box.closest('.checklist');
      if (list) {
        const all = list.querySelectorAll('.check-box');
        const done = list.querySelectorAll('.check-box.done');
        if (all.length === done.length) {
          const msg = list.parentElement.querySelector('.complete-msg');
          if (msg) msg.style.display = 'block';
        }
      }
    });
  });
}

// ============================================
// Memory Bars Animation
// ============================================
function initMemBars() {
  const observer = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.querySelectorAll('.mem-bar-fill').forEach(bar => bar.classList.add('go'));
      }
    });
  }, { threshold: 0.5 });
  
  document.querySelectorAll('.mem-bars').forEach(m => observer.observe(m));
}

// ============================================
// Storage Utilities
// ============================================
const Storage = {
  get(key, defaultValue = null) {
    try {
      const item = localStorage.getItem(`gpu-learning-${key}`);
      return item ? JSON.parse(item) : defaultValue;
    } catch {
      return defaultValue;
    }
  },
  
  set(key, value) {
    try {
      localStorage.setItem(`gpu-learning-${key}`, JSON.stringify(value));
    } catch {
      console.warn('localStorage unavailable');
    }
  }
};

// ============================================
// Progress Tracking for Lessons
// ============================================
class ProgressTracker {
  constructor(lessonId) {
    this.lessonId = lessonId;
    this.completed = new Set(Storage.get(`progress-${lessonId}`, []));
    this.init();
  }
  
  init() {
    this.updateUI();
    
    document.querySelectorAll('.progress__check, .progress-check__box').forEach(check => {
      check.addEventListener('click', () => this.toggle(check.dataset.section));
    });
    
    document.querySelectorAll('.nav__dot').forEach(dot => {
      dot.addEventListener('click', () => {
        const section = dot.dataset.section;
        document.getElementById(`section-${section}`)?.scrollIntoView({ behavior: 'smooth' });
      });
    });
  }
  
  toggle(sectionId) {
    if (this.completed.has(sectionId)) {
      this.completed.delete(sectionId);
    } else {
      this.completed.add(sectionId);
    }
    this.save();
    this.updateUI();
  }
  
  save() {
    Storage.set(`progress-${this.lessonId}`, [...this.completed]);
  }
  
  updateUI() {
    document.querySelectorAll('.progress__check, .progress-check__box').forEach(check => {
      const section = check.dataset.section;
      const isCompleted = this.completed.has(section);
      check.classList.toggle('progress__check--completed', isCompleted);
      check.classList.toggle('completed', isCompleted);
    });
    
    document.querySelectorAll('.nav__dot').forEach(dot => {
      const section = dot.dataset.section;
      dot.classList.toggle('nav__dot--completed', this.completed.has(section));
    });
  }
  
  observeSections() {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const sectionId = entry.target.id.replace('section-', '');
          document.querySelectorAll('.nav__dot').forEach(d => d.classList.remove('nav__dot--active'));
          document.querySelector(`.nav__dot[data-section="${sectionId}"]`)?.classList.add('nav__dot--active');
        }
      });
    }, { threshold: 0.3 });

    document.querySelectorAll('.section[id^="section-"]').forEach(section => {
      observer.observe(section);
    });
  }
}

// ============================================
// Live Computation Component
// ============================================
class LiveComputation {
  constructor(options) {
    this.inputs = options.inputs.map(id => document.getElementById(id));
    this.compute = options.compute;
    this.render = options.render;
    this.init();
  }
  
  init() {
    this.inputs.forEach(input => {
      if (input) {
        input.addEventListener('input', () => this.update());
      }
    });
    this.update();
  }
  
  getValues() {
    return this.inputs.map(input => parseFloat(input?.value) || 0);
  }
  
  update() {
    const values = this.getValues();
    const result = this.compute(values);
    this.render(result, values);
  }
}

// ============================================
// Softmax Visualization
// ============================================
class SoftmaxViz {
  constructor(options) {
    this.inputIds = options.inputs;
    this.barIds = options.bars;
    this.valueIds = options.values;
    this.init();
  }
  
  init() {
    this.inputIds.forEach(id => {
      const input = document.getElementById(id);
      if (input) {
        input.addEventListener('input', () => this.update());
      }
    });
    this.update();
  }
  
  update() {
    const scores = this.inputIds.map(id => {
      const input = document.getElementById(id);
      return parseFloat(input?.value) || 0;
    });
    
    // Stable softmax
    const maxScore = Math.max(...scores);
    const exps = scores.map(s => Math.exp(s - maxScore));
    const sumExp = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map(e => e / sumExp);
    
    probs.forEach((p, i) => {
      const bar = document.getElementById(this.barIds[i]);
      const value = document.getElementById(this.valueIds[i]);
      if (bar) bar.style.height = `${p * 130}px`;
      if (value) value.textContent = p.toFixed(3);
    });
    
    return probs;
  }
}

// ============================================
// Online Softmax Simulation
// ============================================
class OnlineSoftmaxSim {
  constructor(options) {
    this.displayEl = document.getElementById(options.display);
    this.maxEl = document.getElementById(options.maxDisplay);
    this.sumEl = document.getElementById(options.sumDisplay);
    this.countEl = document.getElementById(options.countDisplay);
    this.insightEl = document.getElementById(options.insightDisplay);
    this.addBtn = document.getElementById(options.addBtn);
    this.resetBtn = document.getElementById(options.resetBtn);
    
    this.reset();
    this.init();
  }
  
  init() {
    if (this.addBtn) {
      this.addBtn.addEventListener('click', () => this.addBlock());
    }
    if (this.resetBtn) {
      this.resetBtn.addEventListener('click', () => this.reset());
    }
  }
  
  reset() {
    this.state = { m: -Infinity, l: 0, count: 0, values: [] };
    this.updateUI();
    if (this.displayEl) {
      this.displayEl.innerHTML = '<span class="text-muted text-small">Values will appear here...</span>';
    }
    if (this.insightEl) {
      this.insightEl.textContent = 'Click "Add Random Block" to start the simulation.';
    }
  }
  
  addBlock() {
    const block = Array.from({length: 4}, () => (Math.random() * 6 - 2).toFixed(1));
    const blockNums = block.map(Number);
    
    const m_block = Math.max(...blockNums);
    const m_new = Math.max(this.state.m, m_block);
    
    let l_new;
    if (this.state.m === -Infinity) {
      l_new = 0;
    } else {
      l_new = this.state.l * Math.exp(this.state.m - m_new);
    }
    l_new += blockNums.reduce((sum, x) => sum + Math.exp(x - m_new), 0);
    
    const oldMax = this.state.m;
    this.state.m = m_new;
    this.state.l = l_new;
    this.state.count += 4;
    this.state.values.push(...block);
    
    this.updateUI();
    this.renderTokens(block);
    this.updateInsight(oldMax, m_block, m_new, l_new);
  }
  
  updateUI() {
    if (this.maxEl) {
      this.maxEl.textContent = this.state.m === -Infinity ? '-∞' : this.state.m.toFixed(2);
    }
    if (this.sumEl) {
      this.sumEl.textContent = this.state.l.toFixed(2);
    }
    if (this.countEl) {
      this.countEl.textContent = this.state.count;
    }
  }
  
  renderTokens(block) {
    if (!this.displayEl) return;
    
    if (this.state.values.length === block.length) {
      this.displayEl.innerHTML = '';
    }
    
    block.forEach((v, i) => {
      setTimeout(() => {
        const token = document.createElement('span');
        token.className = 'stream__token';
        token.textContent = v;
        this.displayEl.appendChild(token);
        setTimeout(() => token.classList.add('stream__token--processed'), 200);
      }, i * 100);
    });
  }
  
  updateInsight(oldMax, m_block, m_new, l_new) {
    if (!this.insightEl) return;
    
    if (oldMax !== -Infinity && m_block > oldMax) {
      this.insightEl.textContent = 
        `New max found (${m_block.toFixed(1)} > ${oldMax.toFixed(1)})! Old accumulator rescaled by exp(${oldMax.toFixed(1)} - ${m_new.toFixed(1)}) = ${Math.exp(oldMax - m_new).toFixed(4)}`;
    } else if (oldMax === -Infinity) {
      this.insightEl.textContent = 
        `First block processed. Max = ${m_new.toFixed(2)}, Sum of exp = ${l_new.toFixed(2)}`;
    } else {
      this.insightEl.textContent = 
        `Block max (${m_block.toFixed(1)}) ≤ current max (${m_new.toFixed(1)}). No rescaling needed, just accumulate.`;
    }
  }
}

// ============================================
// FP Bit Toggle
// ============================================
class FPBitToggle {
  constructor(container) {
    this.container = typeof container === 'string' 
      ? document.querySelector(container) 
      : container;
    this.init();
  }
  
  init() {
    this.container?.querySelectorAll('.fp-bit').forEach(bit => {
      bit.addEventListener('click', () => {
        bit.textContent = bit.textContent === '0' ? '1' : '0';
      });
    });
  }
}

// ============================================
// Export for use
// ============================================
window.GPULearning = {
  Storage,
  ProgressTracker,
  LiveComputation,
  SoftmaxViz,
  OnlineSoftmaxSim,
  FPBitToggle
};
