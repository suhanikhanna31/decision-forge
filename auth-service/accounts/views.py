from django.conf import settings
from rest_framework import generics, permissions
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.views import TokenObtainPairView

from .serializers import (
    DecisionForgeTokenObtainPairSerializer,
    RegisterSerializer,
    UserSerializer,
)


class RegisterView(generics.CreateAPIView):
    serializer_class = RegisterSerializer
    permission_classes = [permissions.AllowAny]


class LoginView(TokenObtainPairView):
    """POST username/password -> {access, refresh}. Access token carries
    role/organization claims for downstream services."""

    serializer_class = DecisionForgeTokenObtainPairSerializer
    permission_classes = [permissions.AllowAny]


class MeView(generics.RetrieveAPIView):
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_object(self):
        return self.request.user


class ServiceConfigView(APIView):
    """Lets the React app bootstrap itself with a single call instead of
    hardcoding every service URL."""

    permission_classes = [permissions.AllowAny]

    def get(self, request):
        return Response(
            {
                "ml_service_url": settings.ML_SERVICE_URL,
                "audit_service_url": settings.AUDIT_SERVICE_URL,
            }
        )
