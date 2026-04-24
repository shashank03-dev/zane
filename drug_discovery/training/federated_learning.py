"""
Federated Learning Orchestrator

Uses Flower (flwr) to manage a network of clients training drug discovery models.
Includes custom strategies for handling asynchronous node dropouts.
"""

import logging

import flwr as fl
from flwr.common import Metrics, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

logger = logging.getLogger(__name__)


class RobustFedAvg(FedAvg):
    """Custom FedAvg strategy that handles asynchronous node dropouts."""

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, fl.common.FitRes]],
        failures: list[tuple[ClientProxy, fl.common.FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Handle failures and dropouts during aggregation."""
        if failures:
            logger.warning(f"Round {server_round}: {len(failures)} clients failed/dropped out.")

        # If too many failed, we might want to skip this round or use a different strategy
        if len(results) < self.min_fit_clients:
            logger.error(f"Insufficient results for round {server_round}: {len(results)}/{self.min_fit_clients}")
            return None, {}

        return super().aggregate_fit(server_round, results, failures)


class FederatedServer:
    """Orchestrates the federated learning process."""

    def __init__(self, min_clients: int = 2, num_rounds: int = 3):
        self.strategy = RobustFedAvg(
            min_fit_clients=min_clients,
            min_available_clients=min_clients,
            evaluate_metrics_aggregation_fn=self.weighted_average,
        )
        self.num_rounds = num_rounds

    def weighted_average(self, metrics: list[tuple[int, Metrics]]) -> Metrics:
        """Aggregate metrics from multiple clients."""
        # Multiply accuracy of each client by number of examples used
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        # Aggregate and return custom metric (weighted average)
        return {"accuracy": sum(accuracies) / sum(examples)}

    def start_server(self, server_address: str = "0.0.0.0:8080"):
        """Start the Flower server."""
        logger.info(f"Starting Federated Learning server on {server_address}")
        fl.server.start_server(
            server_address=server_address,
            config=fl.server.ServerConfig(num_rounds=self.num_rounds),
            strategy=self.strategy,
        )
