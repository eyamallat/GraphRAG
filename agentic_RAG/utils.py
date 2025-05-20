from langchain.schema.document import Document
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

def create_documents(candidates):
    documents = []
    for c in candidates:
        content = f"""
            Name: {c.get('name')}
            Title: {c.get('profile_title')}
            Experience: {c.get('years_of_experience')} years
            Resume Summary: {c.get('resume')}
            Education: {format_educations(c.get('educations',[]))}
            Experiences: {format_experiences(c.get('experiences',[]))}
            Languages: {format_langues(c.get('langues',[]))}
            Skills: {format_skills(c.get('skills',[]))}
            Certifications: {format_educations(c.get('educations',[]))}
            Projects: {format_projects(c.get('academic_projects',[]))}
            Missions: {format_missions(c.get('candidat_missions',[]))}
        
        """
        metadata = {
            "id": c.get("id"),
            "name": c.get("name"),
            "lastName": c.get("lastName"),
            "email": c.get("email"),
            "phone_number": c.get("phone_number"),
            "address": c.get("address"),
            "linkedin": c.get("linkedin"),
            "github": c.get("github"),
            "twitter": c.get("twitter"),
            "picture": c.get("picture"),
            "resume_file": c.get("resume_file"),
            "date_of_birth": c.get("date_of_birth"),
            "completude": c.get("completude"),
            "etat": c.get("etat"),
            "score": c.get("score"),
            "mobilite": c.get("mobilite"),
            "pays": c.get("pays"),
            "travail": c.get("travail"),
            "preavis_dispo": c.get("preavis_dispo"),
            "age": c.get("age"),
            "statut": c.get("statut"),
            "genre": c.get("genre"),
            "tjm_std": c.get("tjm_std"),
            "tjm_min": c.get("tjm_min"),
        }

        documents.append(Document(page_content=content.strip(), metadata=clean_metadata(metadata)))
    return documents
