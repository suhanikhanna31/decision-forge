from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    """
    Extends Django's built-in user with a role used across the
    DecisionForge platform (ml-service and audit-service trust the JWT
    issued here and read this role claim to gate access).
    """

    class Role(models.TextChoices):
        ADMIN = "admin", "Admin"
        ANALYST = "analyst", "Analyst"
        VIEWER = "viewer", "Viewer"

    role = models.CharField(max_length=20, choices=Role.choices, default=Role.VIEWER)
    organization = models.CharField(max_length=255, blank=True, default="")

    def __str__(self):
        return f"{self.username} ({self.role})"
