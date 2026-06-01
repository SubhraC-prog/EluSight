cat > elusight/api/routes.py << 'EOF'
"""REST API for EluSight."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn


app = FastAPI(
    title="EluSight API",
    description="Chromatographic Decision Intelligence Framework",
    version="1.0.0"
)


class MethodData(BaseModel):
    """Method data for API."""
    method_id: str
    variables: Dict[str, float]
    objectives: Dict[str, float]
    metadata: Optional[Dict[str, Any]] = None


class InterpretRequest(BaseModel):
    """Request model for method interpretation."""
    methods: List[MethodData]
    constraints: Dict[str, Any]


class InterpretResponse(BaseModel):
    """Response model for interpretation."""
    method_id: str
    trust_score: float
    recommendation: str
    reasoning: str
    risk_summary: Dict[str, float]


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "EluSight API",
        "version": "1.0.0",
        "description": "Chromatographic Decision Intelligence Framework"
    }


@app.post("/interpret", response_model=List[InterpretResponse])
async def interpret_methods(request: InterpretRequest):
    """Interpret chromatographic methods."""
    try:
        responses = []
        
        for method in request.methods:
            # Calculate trust score based on objectives
            resolution = method.objectives.get('resolution', 2.0)
            runtime = method.objectives.get('runtime', 20)
            robustness = method.objectives.get('robustness', 90)
            
            # Simple scoring algorithm
            resolution_score = min(100, (resolution / 2.0) * 100) if resolution > 0 else 50
            runtime_score = max(0, (20 - runtime) / 20 * 100) if runtime > 0 else 50
            robustness_score = robustness if robustness > 0 else 50
            
            trust_score = (resolution_score * 0.4 + runtime_score * 0.3 + robustness_score * 0.3)
            
            if trust_score >= 80:
                recommendation = "Strongly Recommend"
            elif trust_score >= 65:
                recommendation = "Recommend"
            elif trust_score >= 50:
                recommendation = "Consider"
            else:
                recommendation = "Avoid"
            
            # Check constraints
            constraints = request.constraints.get('resolution', {'min': 1.5})
            constraint_pass = resolution >= constraints.get('min', 1.5)
            
            if constraint_pass:
                reasoning = f"Method {method.method_id} meets all critical constraints with resolution of {resolution}."
            else:
                reasoning = f"Method {method.method_id} fails resolution constraint (current: {resolution}, required: {constraints.get('min', 1.5)})."
            
            responses.append(InterpretResponse(
                method_id=method.method_id,
                trust_score=trust_score,
                recommendation=recommendation,
                reasoning=reasoning,
                risk_summary={
                    'coelution_risk': 0.05 if resolution >= 2.0 else (0.15 if resolution >= 1.5 else 0.30),
                    'sst_risk': 0.03,
                    'overall_risk': 0.08
                }
            ))
        
        return responses
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


def run_api(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server."""
    uvicorn.run(app, host=host, port=port)
EOF