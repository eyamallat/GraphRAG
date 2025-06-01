import networkx as nx
from knowledge_graph.utils import  sanitize_id,compute_period

def create_knowledge_graph(candidates):
    """Create a directed knowledge graph from candidate data."""
    G = nx.DiGraph()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue

        person_name = (candidate.get('name', 'Unknown Person')).lower()
        cid = candidate.get('id', '0')
        if not G.has_node(cid):
            G.add_node(cid, 
                    label=person_name,
                    title=person_name, 
                    name=person_name,
                    years_of_experience=candidate.get('years_of_experience', 'N/A'),
                    Email=candidate.get('email', 'N/A'),
                    group="person",
                    id=cid,
                )

        #Process Certifications
        for cert in candidate.get('certifications', []):
            if not isinstance(cert, dict):
                continue
            cert_name = cert.get('name', "Unknown certif").lower()
            issuing_organization = cert.get("issuing_organization","Unknown issuing organization").lower()
            cert_id = sanitize_id(cert,'certification')
            issue_date=cert.get("issue_date","unknown issue date")
            if not G.has_node(cert_id):
                G.add_node(cert_id,
                       label=cert_name,
                       issuing_organization=issuing_organization,
                       issue_date=issue_date,
                       name=cert_name,
                       title=cert_name,
                       group="certification")
            G.add_edge(cid, cert_id, relationship="CERTIFIED_AT")
        # Process Educations
        for edu in candidate.get('educations', []):
            if not isinstance(edu, dict):
                continue  
            univ_name = edu.get('institution_name', 'Unknown University').lower()
            univ_id = sanitize_id(edu.get('id','0'), 'univ')

            if not G.has_node(univ_id):
                G.add_node(univ_id,
                           label=univ_name,
                           name=univ_name,
                           title=univ_name,
                           group="university")

            degree_name = edu.get('degree', 'Unknown Degree').lower()
            degree_id = sanitize_id(degree_name, 'degree')

            if not G.has_node(degree_id):
                G.add_node(degree_id, 
                       label=degree_name,
                       title=degree_name,
                       graduation_date= edu.get('graduation_date', 'N/A'),
                       group="degree")

            # Create relationships
            G.add_edge(cid, univ_id, relationship="STUDIED_AT")
            G.add_edge(univ_id, degree_id, relationship="OFFERS")
            G.add_edge(cid, degree_id, relationship="OBTAINED")

        # Process Experiences
        for exp in candidate.get('experiences', []):
            if not isinstance(exp, dict):
                continue  

            company_name = exp.get('company_name', 'Unknown Company').lower()
            company_id = sanitize_id(company_name,'company')
            company_country=exp.get('country', 'Unknown Country').lower()

            if not G.has_node(company_id):
                G.add_node(company_id,
                           label=company_name,
                           title=company_name,
                           group="company",
                           name=company_name,
                           country=company_country)

            role_name = exp.get('role', 'Unknown Role').lower()
            role_id = sanitize_id(role_name,'role')
            
            if not G.has_node(role_id):
                G.add_node(role_id, 
                       label=role_name,
                       title=role_name,
                       name=role_name,
                       Period= compute_period(exp.get('start_date', ''),exp.get('end_date', '')),
                       Achievements=exp.get('achievements', 'N/A'),
                       group="role")

            # Create relationships
            G.add_edge(cid, company_id, relationship="WORKED_AT")
            G.add_edge(company_id, role_id, relationship="OFFERED")
            G.add_edge(cid, role_id, relationship="PERFORMED")

        # Process Skills
        for skill in candidate.get('skills', []):
            if not isinstance(skill, dict):
                continue
            
            skill_name = skill.get("name", "Unknown Skill").lower()
            skill_type = skill.get("skill_type", "Unknown Type").lower()
            skill_level = skill.get("level", "Unknown Level").lower()
            skill_id = sanitize_id(skill_name,'skill') 
            
            if not G.has_node(skill_id):          
                G.add_node(skill_id,
                       label=skill_name, 
                       title=skill_name,
                       Type=skill_type,
                       Level=skill_level,
                       name=skill_name,
                       group="skill")
            G.add_edge(cid, skill_id, relationship="HAS_SKILL")

        # Process Languages
        for lang in candidate.get('langues', []):
            if not isinstance(lang, dict):
                continue
            lang_name = lang.get('language', 'Unknown Language').lower()
            proficiency = lang.get('proficiency', 'Unknown Level').lower()
            lang_id = sanitize_id(lang_name, 'language')
            
            if not G.has_node(lang_id):
                G.add_node(lang_id,
                       label=lang_name,
                       proficiency=proficiency,
                       name=lang_name,
                       title=lang_name,
                       group="language")
            G.add_edge(cid, lang_id, relationship="SPEAKS")

        # Academic Projects
        for proj in candidate.get('academic_projects', []):
            if not isinstance(proj, dict):
                continue
            proj_title = proj.get('project_title', 'Unknown Project')
            proj_desc = proj.get('project_description', '')
            proj_id = sanitize_id(proj_title,'project')
            completion_date=proj.get('completion_date','Unknown Date')
            if not G.has_node(proj_id):
                G.add_node(proj_id,
                       label=proj_title,
                       title=proj_title,
                       name=proj_title,
                       description=proj_desc,
                       completion_date=completion_date,
                       group="project")
            G.add_edge(cid, proj_id, relationship="WORKED_ON")

        # Candidate Missions
        for mission in candidate.get('candidat_missions', []):
            if not isinstance(mission, dict):
                continue
            mission_name = mission.get('name', 'Unknown Mission').lower()
            company_name = mission.get('company', 'Unknown Company').lower()
            mission_id = sanitize_id(mission_name,'mission')
            if not G.has_node(mission_id):
                G.add_node(mission_id,
                       label=mission_name,
                       name=mission_name,
                       title=mission_name,
                       description=mission.get('description', ''),
                       Company= company_name,
                       group="mission")
            G.add_edge(cid, mission_id, relationship="ASSIGNED_TO")

    return G