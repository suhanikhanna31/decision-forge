from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as DjangoUserAdmin

from .models import User


@admin.register(User)
class UserAdmin(DjangoUserAdmin):
    fieldsets = DjangoUserAdmin.fieldsets + (
        ("DecisionForge", {"fields": ("role", "organization")}),
    )
    list_display = ("username", "email", "role", "organization", "is_staff")
    list_filter = ("role", "is_staff", "is_active")
