.PHONY: setup run scan dashboard test logs down clean

setup:
	@test -f .env || cp .env.example .env
	docker compose up -d db
	@echo "Waiting for database..."
	@until docker compose exec db pg_isready -U weather -d weatheredge > /dev/null 2>&1; do sleep 1; done
	poetry run alembic upgrade head
	@echo "Setup complete."

run:
	docker compose up -d

scan:
	poetry run weather-edge scan

dashboard:
	poetry run streamlit run src/monitoring/dashboard.py

test:
	poetry run pytest

logs:
	docker compose logs -f app

down:
	docker compose down

clean:
	docker compose down -v
