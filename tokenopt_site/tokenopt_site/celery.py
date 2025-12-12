# tokenopt_site/celery.py

import os
from celery import Celery

# Imposta il modulo settings di Django di default per Celery
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tokenopt_site.settings")

app = Celery("tokenopt_site")

# Legge la configurazione da Django, tutti i setting che iniziano con CELERY_
app.config_from_object("django.conf:settings", namespace="CELERY")

# Autodiscover: cerca tasks.py dentro tutte le app in INSTALLED_APPS
app.autodiscover_tasks()


@app.task(bind=True)
def debug_task(self):
    print(f"Request: {self.request!r}")
