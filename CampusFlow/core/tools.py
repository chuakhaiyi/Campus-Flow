"""
core/tools.py
Central tool registry for all CampusFlow agents.
Tools are decorated with @is_tool and auto-registered by department.
The LLM calls these tools directly via the agentic loop in core/agent_base.py.

Architecture note: tools are the ONLY layer that touches the repository.
Agents never import repositories directly — they call tools.
"""

from enum import Enum
from typing import Callable
import functools

# Repository is injected at startup via init_tools()
_repo = None
_adapter = None


def init_tools(repo, adapter) -> None:
    """Called once at startup by the dispatcher to inject dependencies."""
    global _repo, _adapter
    _repo = repo
    _adapter = adapter


# ── Tool type & registry ───────────────────────────────────────────────────────

class ToolType(str, Enum):
    READ  = "read"
    WRITE = "write"


_TOOL_REGISTRY: dict[str, dict] = {}


def get_tool_schemas(department: str | None = None) -> list[dict]:
    """Return OpenAI-style tool schemas for one department (+ shared), or all."""
    if department is None:
        return [e["schema"] for e in _TOOL_REGISTRY.values()]
    return [
        e["schema"] for e in _TOOL_REGISTRY.values()
        if e["department"] in (department, "shared")
    ]


def dispatch_tool(name: str, arguments: dict):
    entry = _TOOL_REGISTRY.get(name)
    if entry is None:
        raise ValueError(f"Unknown tool: {name}")
    return entry["fn"](**arguments)


def is_tool(tool_type: ToolType, schema: dict, department: str = "shared"):
    """Decorator — registers a function as a callable tool."""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)
        _TOOL_REGISTRY[fn.__name__] = {
            "fn":         wrapper,
            "tool_type":  tool_type,
            "department": department,
            "schema":     schema,
        }
        return wrapper
    return decorator

# ══════════════════════════════════════════════════════════════════════════════
#  General TOOLS
# ══════════════════════════════════════════════════════════════════════════════

# Valid top-level field names in student_information.json
_STUDENT_FIELDS = {"personal", "academic", "finance", "hostel", "library", "disciplinary"}


def _load_student_db() -> list[dict]:
    """Load the student_information.json from the configured data path."""
    import json, os
    # _adapter carries the data path via self.path (ticket_path from config)
    base = getattr(_adapter, "path", None)
    if base is None:
        return []
    fpath = os.path.join(base, "student_information.json")
    if not os.path.exists(fpath):
        return []
    with open(fpath) as f:
        data = json.load(f)
    return data.get("students", []) if isinstance(data, dict) else []


def _find_student(student_id: str) -> dict | None:
    for s in _load_student_db():
        if s.get("student_id") == student_id:
            return s
    return None


@is_tool(ToolType.READ, department="shared", schema={
    "type": "function",
    "function": {
        "name": "get_student_info",
        "description": (
            "Fetch a student's profile from student_information.json. "
            "Use the `fields` parameter to request only the sections you need "
            "(e.g. [\"academic\", \"finance\"]) and avoid flooding the LLM context "
            "with irrelevant data. Omit `fields` to retrieve the full profile. "
            "Available sections: personal, academic, finance, hostel, library, disciplinary."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "student_id": {
                    "type": "string",
                    "description": "Student ID, e.g. 'STU-2024-0001'",
                },
                "fields": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": sorted(_STUDENT_FIELDS),
                    },
                    "description": (
                        "Subset of profile sections to return. "
                        "Omit or pass [] to get the full profile."
                    ),
                },
            },
            "required": ["student_id"],
        },
    },
})
def get_student_info(student_id: str, fields: list[str] | None = None) -> dict:
    student = _find_student(student_id)
    if student is None:
        return {
            "found": False,
            "student_id": student_id,
            "message": f"No student record found for ID '{student_id}'.",
        }

    # Validate requested fields; silently drop unknown ones
    requested = [f for f in (fields or []) if f in _STUDENT_FIELDS]

    if requested:
        profile = {f: student.get(f) for f in requested}
    else:
        # Return everything except student_id (it's in the wrapper)
        profile = {k: v for k, v in student.items() if k != "student_id"}

    return {
        "found": True,
        "student_id": student_id,
        "fields_returned": requested if requested else list(_STUDENT_FIELDS),
        "profile": profile,
    }


@is_tool(ToolType.READ, department="shared", schema={
    "type": "function",
    "function": {
        "name": "get_student_summary",
        "description": (
            "Lightweight identity check for a student. Returns only the key fields "
            "needed to verify who the student is and their current academic standing: "
            "name, programme, faculty, year, semester, status, and CGPA. "
            "Prefer this over get_student_info when you only need to confirm identity "
            "or give the orchestrator enough context to route a request."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "student_id": {
                    "type": "string",
                    "description": "Student ID, e.g. 'STU-2024-0001'",
                },
            },
            "required": ["student_id"],
        },
    },
})
def get_student_summary(student_id: str) -> dict:
    student = _find_student(student_id)
    if student is None:
        return {
            "found": False,
            "student_id": student_id,
            "message": f"No student record found for ID '{student_id}'.",
        }

    personal  = student.get("personal", {})
    academic  = student.get("academic", {})

    return {
        "found": True,
        "student_id": student_id,
        "name":        personal.get("name"),
        "email":       personal.get("email"),
        "programme":   academic.get("programme"),
        "faculty":     academic.get("faculty"),
        "year":        academic.get("year"),
        "semester":    academic.get("semester"),
        "status":      academic.get("status"),
        "cgpa":        academic.get("cgpa"),          # None for Year-1 Sem-1
        "intake":      academic.get("intake"),
        "expected_graduation": academic.get("expected_graduation"),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  accommodation TOOLS
# ══════════════════════════════════════════════════════════════════════════════

@is_tool(ToolType.READ, department="accommodation", schema={
    "type": "function",
    "function": {
        "name": "get_available_rooms",
        "description": "Fetch currently available hostel rooms filtered by gender, type, and budget.",
        "parameters": {
            "type": "object",
            "properties": {
                "gender":               {"type": "string", "enum": ["male", "female", "any"]},
                "room_type":            {"type": "string", "description": "'single' or 'double'"},
                "max_monthly_rate_myr": {"type": "number", "description": "Budget cap in MYR"},
            },
            "required": [],
        },
    },
})
def get_available_rooms(gender: str = "any", room_type: str | None = None,
                        max_monthly_rate_myr: float | None = None) -> dict:
    filters = {"status": "available"}
    if gender != "any":
        filters["gender"] = gender
    if room_type:
        filters["room_type"] = room_type
    if max_monthly_rate_myr is not None:
        filters["max_monthly_rate_myr"] = max_monthly_rate_myr
    rooms = _adapter.get_rooms(filters)
    return {"rooms": rooms, "count": len(rooms)}


@is_tool(ToolType.READ, department="accommodation", schema={
    "type": "function",
    "function": {
        "name": "get_accommodation_rules",
        "description": "Get hostel eligibility rules, deposit amount, minimum stay, and required documents.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
})
def get_accommodation_rules() -> dict:
    data = _adapter.read_dept_data("accommodation")
    return data.get("rules", {}) if isinstance(data, dict) else {}


@is_tool(ToolType.READ, department="accommodation", schema={
    "type": "function",
    "function": {
        "name": "get_accommodation_facilities",
        "description": "Get hostel common facilities (cafeteria, laundry, surau, etc.) and policies.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
})
def get_accommodation_facilities() -> dict:
    data = _adapter.read_dept_data("accommodation")
    return data.get("facilities", {}) if isinstance(data, dict) else {}


@is_tool(ToolType.READ, department="accommodation", schema={
    "type": "function",
    "function": {
        "name": "get_room_details",
        "description": "Get full details for a specific room by its room ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "room_id": {"type": "string", "description": "e.g. 'KM-A-101'"},
            },
            "required": ["room_id"],
        },
    },
})
def get_room_details(room_id: str) -> dict:
    data = _adapter.read_dept_data("accommodation")
    rooms = data.get("rooms", []) if isinstance(data, dict) else []
    for room in rooms:
        if room.get("room_id") == room_id:
            return {"found": True, "room": room}
    return {"found": False, "room_id": room_id, "message": "Room not found."}


@is_tool(ToolType.WRITE, department="accommodation", schema={
    "type": "function",
    "function": {
        "name": "submit_accommodation_application",
        "description": "Submit a hostel room application. Only call when student EXPLICITLY asks to apply or book.",
        "parameters": {
            "type": "object",
            "properties": {
                "student_id":       {"type": "string"},
                "student_name":     {"type": "string"},
                "room_id":          {"type": "string"},
                "room_type":        {"type": "string"},
                "monthly_rate_myr": {"type": "number"},
                "additional_notes": {"type": "string"},
            },
            "required": ["student_id", "student_name", "room_id"],
        },
    },
})
def submit_accommodation_application(student_id: str, student_name: str, room_id: str,
                                room_type: str = "", monthly_rate_myr: float = 0,
                                additional_notes: str = "") -> dict:
    ticket = _repo.save_ticket("accommodation", {
        "student_id": student_id, "student_name": student_name,
        "room_id": room_id, "room_type": room_type,
        "monthly_rate_myr": monthly_rate_myr,
        "additional_notes": additional_notes,
        "department": "accommodation", "action": "application_submitted",
    })
    return {
        "success": True,
        "ticket_id": ticket["ticket_id"],
        "message": f"Application submitted for room {room_id}. Ticket: {ticket['ticket_id']}",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ACADEMIC TOOLS
# ══════════════════════════════════════════════════════════════════════════════

@is_tool(ToolType.READ, department="academic", schema={
    "type": "function",
    "function": {
        "name": "get_course_details",
        "description": "Look up a specific course by code. Returns seat availability, schedule, and lecturer.",
        "parameters": {
            "type": "object",
            "properties": {
                "course_code": {"type": "string", "description": "e.g. 'BCS3023'"},
            },
            "required": ["course_code"],
        },
    },
})
def get_course_details(course_code: str) -> dict:
    data = _adapter.read_dept_data("academic")
    courses = data.get("courses", []) if isinstance(data, dict) else []
    for c in courses:
        if c["course_code"].upper() == course_code.upper():
            return {"found": True, "course": c}
    return {"found": False, "course_code": course_code, "message": "Course not found."}


@is_tool(ToolType.READ, department="academic", schema={
    "type": "function",
    "function": {
        "name": "list_all_courses",
        "description": "List all courses, optionally filtered to available seats only.",
        "parameters": {
            "type": "object",
            "properties": {
                "available_only": {"type": "boolean"},
            },
            "required": [],
        },
    },
})
def list_all_courses(available_only: bool = False) -> dict:
    courses = _adapter.get_courses({"available_only": available_only} if available_only else {})
    return {"courses": courses, "count": len(courses)}


@is_tool(ToolType.READ, department="academic", schema={
    "type": "function",
    "function": {
        "name": "get_academic_calendar",
        "description": "Get semester dates, add/drop deadline, exam period, result release, and grade appeal deadline.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
})
def get_academic_calendar() -> dict:
    data = _adapter.read_dept_data("academic")
    return data.get("academic_calendar", {}) if isinstance(data, dict) else {}


@is_tool(ToolType.READ, department="academic", schema={
    "type": "function",
    "function": {
        "name": "get_academic_policies",
        "description": "Get academic policies: GPA thresholds, probation rules, credit transfer limits, dean's list criteria.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
})
def get_academic_policies() -> dict:
    data = _adapter.read_dept_data("academic")
    return data.get("policies", {}) if isinstance(data, dict) else {}


@is_tool(ToolType.WRITE, department="academic", schema={
    "type": "function",
    "function": {
        "name": "submit_academic_request",
        "description": "Submit an academic request ticket (grade appeal, transcript, dean letter, etc.). Only call when student explicitly wants to submit.",
        "parameters": {
            "type": "object",
            "properties": {
                "student_id":   {"type": "string"},
                "student_name": {"type": "string"},
                "request_type": {"type": "string", "description": "e.g. 'grade_appeal', 'transcript_request'"},
                "course_code":  {"type": "string"},
                "details":      {"type": "string"},
            },
            "required": ["student_id", "student_name", "request_type", "details"],
        },
    },
})
def submit_academic_request(student_id: str, student_name: str, request_type: str,
                             details: str, course_code: str = "") -> dict:
    ticket = _repo.save_ticket("academic", {
        "student_id": student_id, "student_name": student_name,
        "request_type": request_type, "course_code": course_code,
        "details": details, "department": "academic",
    })
    return {
        "success": True,
        "ticket_id": ticket["ticket_id"],
        "message": f"Academic request '{request_type}' submitted. Ticket: {ticket['ticket_id']}",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  FINANCE TOOLS
# ══════════════════════════════════════════════════════════════════════════════

@is_tool(ToolType.READ, department="finance", schema={
    "type": "function",
    "function": {
        "name": "get_scholarships_and_bursaries",
        "description": "Fetch available scholarships and bursaries. Pass student's CGPA to filter eligible ones.",
        "parameters": {
            "type": "object",
            "properties": {
                "min_cgpa": {"type": "number", "description": "Student's CGPA to filter eligibility"},
            },
            "required": [],
        },
    },
})
def get_scholarships_and_bursaries(min_cgpa: float | None = None) -> dict:
    data = _adapter.read_dept_data("finance")
    if not isinstance(data, dict):
        return {"scholarships": [], "count": 0}
    results = data.get("scholarships", []) + data.get("bursaries", [])
    if min_cgpa is not None:
        results = [s for s in results if s["eligibility"].get("min_cgpa", 0) <= min_cgpa]
    return {"scholarships": results, "count": len(results)}


@is_tool(ToolType.READ, department="finance", schema={
    "type": "function",
    "function": {
        "name": "get_fee_structure",
        "description": "Get tuition fees per credit hour, registration fee, lab fee, and other charges.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
})
def get_fee_structure() -> dict:
    data = _adapter.read_dept_data("finance")
    return data.get("fee_structure", {}) if isinstance(data, dict) else {}


@is_tool(ToolType.READ, department="finance", schema={
    "type": "function",
    "function": {
        "name": "get_payment_plans",
        "description": "Get instalment options, minimum outstanding balance, and required documents.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
})
def get_payment_plans() -> dict:
    data = _adapter.read_dept_data("finance")
    return data.get("payment_plans", {}) if isinstance(data, dict) else {}


@is_tool(ToolType.READ, department="finance", schema={
    "type": "function",
    "function": {
        "name": "get_refund_policy",
        "description": "Get refund percentages by withdrawal week.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
})
def get_refund_policy() -> dict:
    data = _adapter.read_dept_data("finance")
    return data.get("refund_policy", {}) if isinstance(data, dict) else {}


@is_tool(ToolType.WRITE, department="finance", schema={
    "type": "function",
    "function": {
        "name": "submit_finance_request",
        "description": "Submit a finance request ticket (scholarship application, refund, payment plan). Only call when student explicitly wants to submit.",
        "parameters": {
            "type": "object",
            "properties": {
                "student_id":   {"type": "string"},
                "student_name": {"type": "string"},
                "request_type": {"type": "string"},
                "amount_myr":   {"type": "number"},
                "details":      {"type": "string"},
            },
            "required": ["student_id", "student_name", "request_type", "details"],
        },
    },
})
def submit_finance_request(student_id: str, student_name: str, request_type: str,
                            details: str, amount_myr: float = 0) -> dict:
    ticket = _repo.save_ticket("finance", {
        "student_id": student_id, "student_name": student_name,
        "request_type": request_type, "amount_myr": amount_myr,
        "details": details, "department": "finance",
    })
    return {
        "success": True,
        "ticket_id": ticket["ticket_id"],
        "message": f"Finance request '{request_type}' submitted. Ticket: {ticket['ticket_id']}",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  IT SUPPORT TOOLS
# ══════════════════════════════════════════════════════════════════════════════

@is_tool(ToolType.READ, department="it_support", schema={
    "type": "function",
    "function": {
        "name": "check_system_status",
        "description": "Check live status of campus IT systems (portal, Wi-Fi, email, LMS, etc.).",
        "parameters": {
            "type": "object",
            "properties": {
                "system_name": {"type": "string", "description": "e.g. 'wifi', 'portal', 'email'"},
            },
            "required": [],
        },
    },
})
def check_system_status(system_name: str | None = None) -> dict:
    data = _adapter.read_dept_data("it_support")
    systems = data.get("systems", []) if isinstance(data, dict) else []
    if system_name:
        systems = [s for s in systems if system_name.lower() in s.get("name", "").lower()]
    return {"systems": systems, "count": len(systems)}


@is_tool(ToolType.READ, department="it_support", schema={
    "type": "function",
    "function": {
        "name": "get_known_it_issues",
        "description": "Get all active unresolved IT issues and their documented workarounds.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
})
def get_known_it_issues() -> dict:
    data = _adapter.read_dept_data("it_support")
    issues = data.get("known_issues", []) if isinstance(data, dict) else []
    active = [i for i in issues if i.get("status") != "resolved"]
    return {"known_issues": active, "count": len(active)}


@is_tool(ToolType.READ, department="it_support", schema={
    "type": "function",
    "function": {
        "name": "get_software_licenses",
        "description": "Check software license availability and seat counts.",
        "parameters": {
            "type": "object",
            "properties": {
                "software_name": {"type": "string", "description": "e.g. 'MATLAB', 'AutoCAD'"},
            },
            "required": [],
        },
    },
})
def get_software_licenses(software_name: str | None = None) -> dict:
    data = _adapter.read_dept_data("it_support")
    licenses = data.get("software_licenses", []) if isinstance(data, dict) else []
    if software_name:
        licenses = [l for l in licenses if software_name.lower() in l.get("software", "").lower()]
    return {"software_licenses": licenses, "count": len(licenses)}


@is_tool(ToolType.WRITE, department="it_support", schema={
    "type": "function",
    "function": {
        "name": "submit_it_ticket",
        "description": "Submit an IT support ticket. Only call when user explicitly wants to raise a ticket.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_id":     {"type": "string"},
                "user_name":   {"type": "string"},
                "issue_type":  {"type": "string"},
                "severity":    {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                "description": {"type": "string"},
                "location":    {"type": "string"},
            },
            "required": ["user_id", "user_name", "issue_type", "severity", "description"],
        },
    },
})
def submit_it_ticket(user_id: str, user_name: str, issue_type: str,
                     severity: str, description: str, location: str = "") -> dict:
    ticket = _repo.save_ticket("it_support", {
        "user_id": user_id, "user_name": user_name,
        "issue_type": issue_type, "severity": severity,
        "description": description, "location": location,
        "department": "it_support",
    })
    return {
        "success": True,
        "ticket_id": ticket["ticket_id"],
        "message": f"IT ticket raised. Ticket: {ticket['ticket_id']}",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  LIBRARY TOOLS
# ══════════════════════════════════════════════════════════════════════════════

@is_tool(ToolType.READ, department="library", schema={
    "type": "function",
    "function": {
        "name": "search_books",
        "description": "Search for books by title keyword or ISBN. Returns real copy counts and availability.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Title keyword or ISBN"},
            },
            "required": ["query"],
        },
    },
})
def search_books(query: str) -> dict:
    data = _adapter.read_dept_data("library")
    books = data.get("books", []) if isinstance(data, dict) else []
    q = query.lower()
    matches = [b for b in books if q in b.get("title", "").lower() or q in b.get("isbn", "")]
    return {"books": matches, "count": len(matches)}


@is_tool(ToolType.READ, department="library", schema={
    "type": "function",
    "function": {
        "name": "get_borrowing_rules",
        "description": "Get book borrowing limits, loan duration, and renewal rules by user type.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
})
def get_borrowing_rules() -> dict:
    data = _adapter.read_dept_data("library")
    return data.get("borrowing_rules", {}) if isinstance(data, dict) else {}


@is_tool(ToolType.READ, department="library", schema={
    "type": "function",
    "function": {
        "name": "get_study_rooms",
        "description": "Get available study rooms, capacity, features (projector, whiteboard), and booking info.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
})
def get_study_rooms() -> dict:
    data = _adapter.read_dept_data("library")
    rooms = data.get("study_rooms", []) if isinstance(data, dict) else []
    return {"study_rooms": rooms, "count": len(rooms)}


@is_tool(ToolType.READ, department="library", schema={
    "type": "function",
    "function": {
        "name": "get_fine_rates",
        "description": "Get overdue fine rates per day and lost book replacement policy.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
})
def get_fine_rates() -> dict:
    data = _adapter.read_dept_data("library")
    return data.get("fines", {}) if isinstance(data, dict) else {}


@is_tool(ToolType.READ, department="library", schema={
    "type": "function",
    "function": {
        "name": "get_library_databases",
        "description": "Get academic databases (IEEE Xplore, Scopus, etc.) and their on-campus/VPN access requirements.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
})
def get_library_databases() -> dict:
    data = _adapter.read_dept_data("library")
    dbs = data.get("databases", []) if isinstance(data, dict) else []
    return {"databases": dbs, "count": len(dbs)}


@is_tool(ToolType.WRITE, department="library", schema={
    "type": "function",
    "function": {
        "name": "submit_library_request",
        "description": "Submit a library request ticket (book reservation, study room booking, fine dispute). Only call when user explicitly wants to submit.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_id":      {"type": "string"},
                "user_name":    {"type": "string"},
                "request_type": {"type": "string"},
                "details":      {"type": "string"},
            },
            "required": ["user_id", "user_name", "request_type", "details"],
        },
    },
})
def submit_library_request(user_id: str, user_name: str, request_type: str, details: str) -> dict:
    ticket = _repo.save_ticket("library", {
        "user_id": user_id, "user_name": user_name,
        "request_type": request_type, "details": details,
        "department": "library",
    })
    return {
        "success": True,
        "ticket_id": ticket["ticket_id"],
        "message": f"Library request submitted. Ticket: {ticket['ticket_id']}",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAINTENANCE TOOLS
# ══════════════════════════════════════════════════════════════════════════════

@is_tool(ToolType.READ, department="maintenance", schema={
    "type": "function",
    "function": {
        "name": "get_maintenance_staff",
        "description": "Get currently available maintenance staff, optionally filtered by role.",
        "parameters": {
            "type": "object",
            "properties": {
                "role": {"type": "string", "description": "e.g. 'electrician', 'plumber', 'HVAC'"},
            },
            "required": [],
        },
    },
})
def get_maintenance_staff(role: str | None = None) -> dict:
    data = _adapter.read_dept_data("maintenance")
    staff = data.get("staff", []) if isinstance(data, dict) else []
    available = [s for s in staff if s.get("available")]
    if role:
        available = [s for s in available if role.lower() in s.get("role", "").lower()]
    return {"available_staff": available, "count": len(available)}


@is_tool(ToolType.READ, department="maintenance", schema={
    "type": "function",
    "function": {
        "name": "get_equipment_registry",
        "description": "Look up campus equipment/assets by location or asset ID to check status and warranty.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "asset_id": {"type": "string"},
            },
            "required": [],
        },
    },
})
def get_equipment_registry(location: str | None = None, asset_id: str | None = None) -> dict:
    data = _adapter.read_dept_data("maintenance")
    equipment = data.get("equipment_registry", []) if isinstance(data, dict) else []
    if asset_id:
        equipment = [e for e in equipment if e.get("asset_id") == asset_id]
    if location:
        equipment = [e for e in equipment if location.lower() in e.get("location", "").lower()]
    return {"equipment": equipment, "count": len(equipment)}


@is_tool(ToolType.READ, department="maintenance", schema={
    "type": "function",
    "function": {
        "name": "get_maintenance_sla",
        "description": "Get SLA response time targets by priority level (critical/high/medium/low).",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
})
def get_maintenance_sla() -> dict:
    data = _adapter.read_dept_data("maintenance")
    return data.get("sla", {}) if isinstance(data, dict) else {}


@is_tool(ToolType.WRITE, department="maintenance", schema={
    "type": "function",
    "function": {
        "name": "submit_maintenance_request",
        "description": "Submit a maintenance/repair ticket. Only call when user wants to formally log the issue.",
        "parameters": {
            "type": "object",
            "properties": {
                "requester_id":      {"type": "string"},
                "requester_name":    {"type": "string"},
                "category":          {"type": "string", "description": "aircon|plumbing|electrical|structural|lift|cleaning|furniture|general"},
                "priority":          {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                "location":          {"type": "string"},
                "description":       {"type": "string"},
                "assigned_staff_id": {"type": "string"},
            },
            "required": ["requester_id", "requester_name", "category", "priority", "location", "description"],
        },
    },
})
def submit_maintenance_request(requester_id: str, requester_name: str, category: str,
                                priority: str, location: str, description: str,
                                assigned_staff_id: str = "") -> dict:
    ticket = _repo.save_ticket("maintenance", {
        "requester_id": requester_id, "requester_name": requester_name,
        "category": category, "priority": priority,
        "location": location, "description": description,
        "assigned_staff_id": assigned_staff_id, "department": "maintenance",
    })
    return {
        "success": True,
        "ticket_id": ticket["ticket_id"],
        "message": f"Maintenance request submitted. Ticket: {ticket['ticket_id']}",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PROCUREMENT TOOLS
# ══════════════════════════════════════════════════════════════════════════════

@is_tool(ToolType.READ, department="procurement", schema={
    "type": "function",
    "function": {
        "name": "get_approved_vendors",
        "description": "Get approved vendors, optionally filtered by supply category.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "e.g. 'it_hardware', 'furniture'"},
            },
            "required": [],
        },
    },
})
def get_approved_vendors(category: str | None = None) -> dict:
    data = _adapter.read_dept_data("procurement")
    vendors = data.get("approved_vendors", []) if isinstance(data, dict) else []
    if category:
        vendors = [v for v in vendors if any(category.lower() in c.lower() for c in v.get("categories", []))]
    return {"vendors": vendors, "count": len(vendors)}


@is_tool(ToolType.READ, department="procurement", schema={
    "type": "function",
    "function": {
        "name": "get_approval_tiers",
        "description": "Get procurement approval tiers and MYR thresholds to determine who must sign off.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
})
def get_approval_tiers() -> dict:
    data = _adapter.read_dept_data("procurement")
    return {"approval_tiers": data.get("approval_tiers", [])} if isinstance(data, dict) else {}


@is_tool(ToolType.READ, department="procurement", schema={
    "type": "function",
    "function": {
        "name": "check_department_budget",
        "description": "Check remaining budget for a department.",
        "parameters": {
            "type": "object",
            "properties": {
                "department_name": {"type": "string"},
            },
            "required": [],
        },
    },
})
def check_department_budget(department_name: str | None = None) -> dict:
    data = _adapter.read_dept_data("procurement")
    budgets = data.get("budget_remaining", {}) if isinstance(data, dict) else {}
    if department_name:
        key = department_name.lower().replace(" ", "_")
        return {"department": department_name, "budget": budgets.get(key, "unknown")}
    return {"all_budgets": budgets}


@is_tool(ToolType.READ, department="procurement", schema={
    "type": "function",
    "function": {
        "name": "get_active_purchase_orders",
        "description": "List active purchase orders to check for duplicate requests.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
})
def get_active_purchase_orders() -> dict:
    data = _adapter.read_dept_data("procurement")
    pos = data.get("active_purchase_orders", []) if isinstance(data, dict) else []
    return {"purchase_orders": pos, "count": len(pos)}


@is_tool(ToolType.WRITE, department="procurement", schema={
    "type": "function",
    "function": {
        "name": "submit_procurement_request",
        "description": "Submit a procurement request ticket. Only call when requester explicitly wants to proceed.",
        "parameters": {
            "type": "object",
            "properties": {
                "requester_id":       {"type": "string"},
                "requester_name":     {"type": "string"},
                "department_name":    {"type": "string"},
                "request_type":       {"type": "string"},
                "items":              {"type": "string"},
                "estimated_cost_myr": {"type": "number"},
                "justification":      {"type": "string"},
            },
            "required": ["requester_id", "requester_name", "department_name",
                         "request_type", "items", "estimated_cost_myr"],
        },
    },
})
def submit_procurement_request(requester_id: str, requester_name: str, department_name: str,
                                request_type: str, items: str, estimated_cost_myr: float,
                                justification: str = "") -> dict:
    ticket = _repo.save_ticket("procurement", {
        "requester_id": requester_id, "requester_name": requester_name,
        "department_name": department_name, "request_type": request_type,
        "items": items, "estimated_cost_myr": estimated_cost_myr,
        "justification": justification, "department": "procurement",
    })
    return {
        "success": True,
        "ticket_id": ticket["ticket_id"],
        "message": f"Procurement request submitted. Ticket: {ticket['ticket_id']}",
    }
