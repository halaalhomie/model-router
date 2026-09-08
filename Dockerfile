# Matches the Python the project is developed and tested on.
FROM python:3.14-slim

# Never write .pyc into the image, and never buffer stdout -- an
# unbuffered stream is the difference between seeing a container's logs
# live and seeing them when it exits.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies before source, so editing a .py file does not invalidate
# the layer that installs them. requirements.txt only -- this is where
# splitting out requirements-dev.txt back in Phase 2 pays off: pytest,
# flake8 and httpx never reach the runtime image.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Run as a non-root user. A container process that does not need to write
# to the filesystem has no business owning it.
RUN useradd --create-home --uid 10001 router \
    && chown -R router:router /app
USER router

EXPOSE 8000

# The API is the default, but every entry point lives in this one image --
# docker-compose.yml runs the persistence consumer from it by overriding
# the command. One image, several roles, no drift between them.
CMD ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
