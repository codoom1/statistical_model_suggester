(function () {
    const STAGES = {
        model: [
            [10, 'Checking your study design…'],
            [34, 'Matching compatible statistical models…'],
            [62, 'Comparing the strongest candidates…'],
            [84, 'Preparing your recommendation…'],
            [92, 'Almost ready…']
        ],
        modelAi: [
            [10, 'Checking your study design…'],
            [32, 'Matching compatible statistical models…'],
            [58, 'Reviewing the verified shortlist with AI…'],
            [82, 'Preparing your recommendation…'],
            [92, 'Almost ready…']
        ],
        questionnaire: [
            [10, 'Analyzing your research goals…'],
            [36, 'Building questionnaire sections…'],
            [68, 'Organizing question types…'],
            [86, 'Preparing your questionnaire…'],
            [92, 'Almost ready…']
        ],
        questionnaireAi: [
            [10, 'Analyzing your research goals…'],
            [34, 'Building questionnaire sections…'],
            [60, 'Generating focused AI questions…'],
            [84, 'Preparing your questionnaire…'],
            [92, 'Almost ready…']
        ]
    };

    function resetProgress(form, overlay) {
        overlay.classList.remove('active');
        overlay.setAttribute('aria-hidden', 'true');
        const submitButton = form.querySelector('button[type="submit"]');
        if (submitButton) {
            submitButton.disabled = false;
        }
    }

    function startProgress(form, overlay) {
        const aiFieldId = form.dataset.progressAiField;
        const aiField = aiFieldId ? document.getElementById(aiFieldId) : null;
        const aiEnabled = Boolean(aiField && aiField.checked);
        const mode = form.dataset.progressMode || 'model';
        const stages = STAGES[`${mode}${aiEnabled ? 'Ai' : ''}`] || STAGES[mode];
        const bar = overlay.querySelector('.progress-bar');
        const status = overlay.querySelector('.generation-progress-status');
        const percent = overlay.querySelector('.generation-progress-percent');
        const submitButton = form.querySelector('button[type="submit"]');
        let stageIndex = 0;

        overlay.classList.add('active');
        overlay.setAttribute('aria-hidden', 'false');
        if (submitButton) {
            submitButton.disabled = true;
        }

        function showStage() {
            const stage = stages[Math.min(stageIndex, stages.length - 1)];
            bar.style.width = `${stage[0]}%`;
            bar.setAttribute('aria-valuenow', String(stage[0]));
            status.textContent = stage[1];
            percent.textContent = `${stage[0]}%`;
            if (stageIndex < stages.length - 1) {
                stageIndex += 1;
            }
        }

        showStage();
        window.setInterval(showStage, 1800);
    }

    document.addEventListener('DOMContentLoaded', function () {
        document.querySelectorAll('[data-generation-progress]').forEach(function (form) {
            const overlay = document.getElementById(form.dataset.progressTarget);
            if (!overlay) {
                return;
            }
            resetProgress(form, overlay);
            form.addEventListener('submit', function (event) {
                window.queueMicrotask(function () {
                    if (!event.defaultPrevented) {
                        startProgress(form, overlay);
                    }
                });
            });
        });
    });

    window.addEventListener('pageshow', function () {
        document.querySelectorAll('[data-generation-progress]').forEach(function (form) {
            const overlay = document.getElementById(form.dataset.progressTarget);
            if (overlay) {
                resetProgress(form, overlay);
            }
        });
    });
}());
