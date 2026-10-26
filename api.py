import asyncio
import json

import numpy as np
import equinox as eqx
import jax.numpy as jnp

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from pydantic import BaseModel
from typing import List


from flowtangent import Aircraft, Settings, JacobianMap, units
from flowtangent.utils import TreePath
from flowtangent.components.wings import (Wing, WingChords, WingSweeps, WingSegment as Segment)
from flowtangent.solve import AnalysisSettings, BatchedAnalysis
from flowtangent.solve.aero import VORJAX, VORJAXSettings

# --- 1. PYDANTIC DATA MODELS ---
# These define the strict schema for incoming JSON payloads

class FlightRegime(BaseModel):
    mach: float
    alpha: float
    beta: float

class WingSegment(BaseModel):
    name: str
    span: float
    taper: float
    sweep: float
    dihedral: float
    twist: float

class MainWing(BaseModel):
    root_chord: float
    root_twist: float
    segments: List[WingSegment]

class Vehicle(BaseModel):
    main_wing: MainWing

class AnalysisPayload(BaseModel):
    flight_regime: FlightRegime
    vehicle: Vehicle

# --- 2. FASTAPI SERVER INITIALIZATION ---

def make_json_serializable(data):
    """
    Recursively strips JAX and NumPy types out of a data structure, 
    converting them to native Python lists and floats.
    """
    if hasattr(data, 'tolist'):
        # Catches BOTH jax.Array and numpy.ndarray seamlessly
        return data.tolist()
    elif hasattr(data, 'item'):
        # Catches JAX/NumPy scalars (e.g., np.float64, jnp.float32)
        return data.item()
    elif isinstance(data, dict):
        # Recursively search through dictionaries
        return {k: make_json_serializable(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        # Recursively search through lists/tuples
        return [make_json_serializable(v) for v in data]
    
    # Base case: strings, native ints/floats, bools, None
    return data

def wing_generator(wing: MainWing):

    segments_json = wing.segments
    segments = []

    span = np.sum([seg.span for seg in segments_json])
    tip_twist = np.sum([seg.twist for seg in segments_json])


    for s_idx, seg in enumerate(segments_json):
        if s_idx == 0:
            new_segment = Segment(
                name=seg.name,
                percent_span_location=0.0,
                root_chord_percent=1.0,
                twist=seg.twist * units.deg,
                dihedral_outboard=seg.dihedral * units.deg,
                sweeps=WingSweeps(leading_edge=seg.sweep * units.deg),
            )
        else:
            new_segment = Segment(
                name=seg.name,
                percent_span_location=segments[-1].span/span,
                root_chord_percent=np.prod([s.taper for s in segments_json[:s_idx]]),
                twist=seg.twist * units.deg,
                dihedral_outboard=seg.dihedral * units.deg,
                sweeps=WingSweeps(leading_edge=seg.sweep * units.deg),
            )
        
        segments.append(new_segment)

        wing = Wing(
            name="API Wing",
            symmetric=True,
            taper = np.prod([s.taper for s in segments_json]),
            chords=WingChords(root=wing.root_chord),
            twists=WingDimensions(root=wing.root_twist),
            spans=WingDimensions(projected=span),
            origin=jnp.array([[0., 0., 0.]]),
        ).update_geometry(calculate_reference_area=True, calculate_wetted_area=True)

        system = Aircraft(name=f"W1_System", areas=wing.areas).add_subcomponent(wing)
        system = eqx.tree_at(lambda s: s.mass_properties.center_of_gravity, system, jnp.array([[0.0, 0.0, 0.0]]))

        return system

def runVORJAX(flight_regime: FlightRegime, system: Aircraft):

    solver=BatchVORJAX()

    mach_path   = TreePath(("freestream", "mach_number"), name="M")
    alpha_path  = TreePath(("aerodynamics", "angles", "alpha"), name="a")
    beta_path   = TreePath(("aerodynamics", "angles", "beta"), name="b")

    lift_path   = TreePath(("aerodynamics", "coefficients", "lift", "total"), name="CL")
    drag_path   = TreePath(("aerodynamics", "coefficients", "drag", "total"), name="CD")
    pitch_path   = TreePath(("aerodynamics", "coefficients", "moments", "pitch"), name="C_m")

    GRAD_MAP = JacobianMap(
        state_inputs=(
            mach_path,
            alpha_path,
            beta_path,
        ),
        state_outputs=(
            lift_path,
            drag_path,
            pitch_path
        )
    )
    
    aero_settings = VORJAX_Settings(vortices=Vortices(n_spanwise=16, n_chordwise=8))
    analysis_settings = AnalysisSettings(
        aerodynamics=aero_settings,
        gradient_map=GRAD_MAP
    )

    settings = Settings(analysis=analysis_settings)

    results = solver.run(
        system=system,
        settings=settings,
        mode="mesh",
        mach=flight_regime.mach,
        alpha=flight_regime.alpha * units.deg,
        beta=flight_regime.beta * units.deg,
        batch_size=1,
        handle="VORJAX_API"
    )
    
    return make_json_serializable(results)


# --- 2. FASTAPI SERVER INITIALIZATION ---
app = FastAPI(title="FT Compute Engine")

# --- 3. API ENDPOINTS ---
@app.post("/solve_mission")
async def solve_mission(payload: AnalysisPayload):
    
    async def process_stream():
        try:
            # Instantiation
            yield json.dumps({"type": "status", "message": "Building wing data structures..."}) + "\n"
            wing = await asyncio.to_thread(wing_generator, payload.vehicle.main_wing)
            
            # VORJAX Compute
            yield json.dumps({"type": "status", "message": "Compiling compute graph ..."}) + "\n"
            coeffs = await asyncio.to_thread(runVORJAX, payload.flight_regime, wing)

            result_payload = {
                "type": "result",
                "coefficients": coeffs
            }
            yield json.dumps(result_payload) + "\n"
            
        except Exception as e:
            # Stream the error safely if something blows up inside the JAX solver
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    # Return the stream immediately. media_type="application/x-ndjson" tells 
    # the client to expect multiple independent JSON objects separated by newlines.
    return StreamingResponse(process_stream(), media_type="application/x-ndjson")

# Run this server from your terminal using:
# uvicorn api:app --reload --port 8000