#!/usr/bin/env python3

import sys

def main():
    print("Verifying EluSight installation...")
    
    try:
        from elusight import __version__
        print(f"✅ EluSight version: {__version__}")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        sys.exit(1)
    
    try:
        from elusight.schemas.base import MethodResult, MethodVariables, MethodObjectives
        from elusight.ingest import Ingestor
        from elusight.trust.engine import TrustEngine
        from elusight.reasoning.engine import ReasoningEngine
        
        print("✅ All core modules imported successfully")
        
        # Test basic functionality
        method = MethodResult(
            method_id="TEST",
            variables=MethodVariables(pH=3.2),
            objectives=MethodObjectives(resolution=2.5)
        )
        print(f"✅ Created test method: {method.method_id}")
        
        trust_engine = TrustEngine()
        reasoning_engine = ReasoningEngine()
        print("✅ Initialized engines")
        
        print("\n🎉 EluSight is ready to use!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()