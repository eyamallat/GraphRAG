from vectorstore.utils import format_educations,format_skills,format_experiences,format_projects,format_langues,format_missions,clean_metadata
from langchain.schema.document import Document


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
