"""
Command-line interface for Drug Discovery Platform
"""

import argparse
import sys
from drug_discovery import DrugDiscoveryPipeline
from drug_discovery.evaluation import ADMETPredictor


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='AI Drug Discovery Platform CLI'
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict properties for a molecule')
    predict_parser.add_argument('smiles', help='SMILES string of the molecule')
    predict_parser.add_argument('--model', default='gnn', choices=['gnn', 'transformer', 'ensemble'])
    predict_parser.add_argument('--checkpoint', help='Path to model checkpoint')

    # ADMET command
    admet_parser = subparsers.add_parser('admet', help='Analyze ADMET properties')
    admet_parser.add_argument('smiles', help='SMILES string of the molecule')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--model', default='gnn', choices=['gnn', 'transformer', 'ensemble'])
    train_parser.add_argument('--epochs', type=int, default=100)
    train_parser.add_argument('--batch-size', type=int, default=32)

    # Collect data command
    collect_parser = subparsers.add_parser('collect', help='Collect molecular data')
    collect_parser.add_argument('--sources', nargs='+', default=['pubchem', 'chembl'])
    collect_parser.add_argument('--limit', type=int, default=1000)

    args = parser.parse_args()

    if args.command == 'predict':
        predict_properties(args)
    elif args.command == 'admet':
        analyze_admet(args)
    elif args.command == 'train':
        train_model(args)
    elif args.command == 'collect':
        collect_data(args)
    else:
        parser.print_help()


def predict_properties(args):
    """Predict molecular properties"""
    print(f"Predicting properties for: {args.smiles}")

    if args.checkpoint:
        pipeline = DrugDiscoveryPipeline(model_type=args.model)
        pipeline.load(args.checkpoint)
        properties = pipeline.predict_properties(args.smiles)

        print("\nPredicted Properties:")
        for key, value in properties.items():
            print(f"  {key}: {value}")
    else:
        print("Error: --checkpoint required for prediction")
        sys.exit(1)


def analyze_admet(args):
    """Analyze ADMET properties"""
    print(f"Analyzing ADMET for: {args.smiles}")

    admet = ADMETPredictor()

    # Lipinski
    lipinski = admet.check_lipinski_rule(args.smiles)
    if lipinski:
        print(f"\nLipinski's Rule: {'PASS' if lipinski['passes'] else 'FAIL'}")
        print(f"Violations: {lipinski['num_violations']}")

    # QED
    qed = admet.calculate_qed(args.smiles)
    if qed:
        print(f"\nDrug-likeness (QED): {qed:.3f}")

    # SA
    sa = admet.calculate_synthetic_accessibility(args.smiles)
    if sa:
        print(f"Synthetic Accessibility: {sa:.2f}/10")


def train_model(args):
    """Train a new model"""
    print(f"Training {args.model} model...")

    pipeline = DrugDiscoveryPipeline(model_type=args.model)

    # Collect data
    data = pipeline.collect_data()

    # Prepare datasets
    train_loader, test_loader = pipeline.prepare_datasets(
        data, batch_size=args.batch_size
    )

    # Train
    history = pipeline.train(train_loader, test_loader, num_epochs=args.epochs)

    # Save
    pipeline.save(f'./checkpoints/{args.model}_model.pt')
    print(f"\nModel saved to ./checkpoints/{args.model}_model.pt")


def collect_data(args):
    """Collect molecular data"""
    print(f"Collecting data from: {', '.join(args.sources)}")

    from drug_discovery.data import DataCollector

    collector = DataCollector()
    data = collector.merge_datasets([])

    for source in args.sources:
        if source == 'pubchem':
            df = collector.collect_from_pubchem(limit=args.limit)
        elif source == 'chembl':
            df = collector.collect_from_chembl(limit=args.limit)
        else:
            continue

        data = collector.merge_datasets([data, df])

    print(f"\nCollected {len(data)} unique molecules")

    # Save
    data.to_csv('./data/collected_data.csv', index=False)
    print("Saved to ./data/collected_data.csv")


if __name__ == '__main__':
    main()
