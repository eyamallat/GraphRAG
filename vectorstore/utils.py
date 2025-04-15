def format_educations(educs):
    return "".join(
        f"- {e['degree']} at {e['institution_name']} (Graduated: {e['graduation_date']})"
        for e in educs
        )

def format_experiences(exps):
    return "".join(
        f"- {e['role']} at {e['company_name']} ({e['start_date']} - {e['end_date'] or 'Present'}): {e['achievements']}"
        for e in exps
        )

def format_langues(langs):
    return "".join(
        f"- {l['language']} ({l['proficiency']})" 
        for l in langs
        )

def format_skills(skills):
    return "".join(
        f"- {s['name']} ({s['level']} - {s['skill_type']})" 
        for s in skills
        )
def format_certifications(certs):
    return "".join(
            f"- {c['name']} from {c['issuing_organization']} (Issued: {c['issue_date']})"
            for c in certs
        )

def format_projects(projects):
    return "".join(
            f"- {p['project_title']}: {p['project_description']} (Completed: {p['completion_date']})"
            for p in projects
        )

def format_missions(missions):
    return "".join(
            f"- {m['name']} at {m['company']} ({m['date']}): {m['description']}"
            for m in missions
        )
def clean_metadata(metadata):
    return {
        k: str(v) if not isinstance(v, (str, int, float, bool)) or v is None else v
        for k, v in metadata.items()
    }