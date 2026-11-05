-- Orthanc : envoie automatiquement chaque nouvelle série FLAIR vers SegBrainMRI.
-- Les résultats renvoyés par SegBrainMRI (Manufacturer = "SegBrainMRI", SEG) sont ignorés,
-- ce qui évite toute boucle.

local TARGET = 'segbrainmri'   -- nom de la modalité déclarée dans "DicomModalities"
local FLAIR_PATTERNS = { 'flair', 'dark.?fluid', 't2.?fl', 'spcir' }

function OnStableSeries(seriesId, tags, metadata)
   local modality = string.upper(tags['Modality'] or '')
   if modality ~= 'MR' or (tags['Manufacturer'] or '') == 'SegBrainMRI' then
      return
   end
   local text = string.lower((tags['SeriesDescription'] or '') .. ' ' ..
                             (tags['ProtocolName'] or '') .. ' ' ..
                             (tags['SequenceName'] or ''))
   for _, pattern in ipairs(FLAIR_PATTERNS) do
      if string.find(text, pattern) then
         print('SegBrainMRI : envoi de la serie FLAIR ' .. seriesId)
         RestApiPost('/modalities/' .. TARGET .. '/store',
                     '{"Resources": ["' .. seriesId .. '"], "Asynchronous": true}')
         return
      end
   end
end
